# Copyright 2025 Individual Contributor: furunding
# Adapted for vLLM HTTP (OpenAI-compatible) API mode.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VLLMHTTPEngine: drop-in replacement for VLLMEngine that calls a remote
vLLM OpenAI-compatible server (launched via `vllm serve`) instead of
loading the model in-process.

Advantages:
  - Avoids Ray Compiled DAG issues (vllm serve handles PP internally)
  - No monkey-patching of LogprobsProcessor
  - Supports continuous batching via the async server
  - Training-side TeacherClient remains unchanged (same tensor interface)

Usage:
  # Start vLLM server separately:
  vllm serve <model> -tp 8 -pp 2 --return-tokens-as-token-ids --max-logprobs 256

  # Then in worker.py:
  engine = VLLMHTTPEngine(server_url="http://localhost:8000",
                          model_name="<model>", n_logprobs=1)
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import requests
import torch


# Regex to extract token_id from "token_id:12345" format
_TOKEN_ID_RE = re.compile(r"^token_id:(\d+)$")


class VLLMHTTPEngine:
    """Drop-in replacement for VLLMEngine that calls a vLLM HTTP server."""

    def __init__(
        self,
        server_url: str,
        model_name: str,
        n_logprobs: int = 0,
        max_concurrent_requests: int = 32,
        timeout: float = 600.0,
    ):
        """
        Args:
            server_url: Base URL of vLLM server, e.g. "http://localhost:8000"
            model_name: Model name as registered in vLLM server
            n_logprobs: Number of top logprobs to return per token
            max_concurrent_requests: Max parallel HTTP requests to server
            timeout: HTTP request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.n_logprobs = n_logprobs
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.session = requests.Session()

        # Verify server is reachable
        self._wait_for_server()

    def _wait_for_server(self, max_retries: int = 60, retry_interval: float = 5.0):
        """Wait until the vLLM server is ready."""
        for i in range(max_retries):
            try:
                resp = self.session.get(
                    f"{self.server_url}/v1/models", timeout=10
                )
                resp.raise_for_status()
                models = resp.json()
                model_ids = [m["id"] for m in models["data"]]
                print(f"[VLLMHTTPEngine] Connected to {self.server_url}")
                print(f"[VLLMHTTPEngine] Available models: {model_ids}")
                if self.model_name not in model_ids:
                    # Use the first available model if exact name not found
                    if model_ids:
                        print(
                            f"[VLLMHTTPEngine] WARNING: model '{self.model_name}' "
                            f"not found, using '{model_ids[0]}'"
                        )
                        self.model_name = model_ids[0]
                return
            except Exception as e:
                if i < max_retries - 1:
                    print(
                        f"[VLLMHTTPEngine] Waiting for server... "
                        f"({i+1}/{max_retries}): {e}"
                    )
                    time.sleep(retry_interval)
                else:
                    raise RuntimeError(
                        f"Cannot connect to vLLM server at {self.server_url} "
                        f"after {max_retries} retries: {e}"
                    )

    def get_topk_logprobs(
        self,
        prompt_token_ids: List[List[int]],
        temperature: float = 0.8,
        max_new_tokens: Union[int, List[int]] = 1,
        only_response: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Same interface as VLLMEngine.get_topk_logprobs.

        Returns:
            responses: list of torch.Tensor (int32) — full token sequence (prompt + generated)
            teacher_topk_logprobs: list of torch.Tensor (float32) — [seq_len, n_logprobs+1]
            teacher_topk_indices: list of torch.Tensor (int32) — [seq_len, n_logprobs+1]
        """
        batch_size = len(prompt_token_ids)

        # Normalize max_new_tokens to a list
        if isinstance(max_new_tokens, list):
            assert len(max_new_tokens) == batch_size
            per_request_max_tokens = max_new_tokens
        else:
            per_request_max_tokens = [max_new_tokens] * batch_size

        # Build request payloads
        payloads = []
        for i in range(batch_size):
            tokens = prompt_token_ids[i]
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()

            payload = {
                "model": self.model_name,
                "prompt": tokens,
                "max_tokens": per_request_max_tokens[i],
                "temperature": temperature,
                "top_p": 0.95,
                "logprobs": self.n_logprobs if self.n_logprobs > 0 else None,
                "echo": False,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            }
            # Only request prompt_logprobs if needed
            if not only_response and self.n_logprobs > 0:
                payload["prompt_logprobs"] = self.n_logprobs

            payloads.append((i, tokens, payload))

        # Send requests concurrently
        results = [None] * batch_size
        with ThreadPoolExecutor(
            max_workers=min(self.max_concurrent_requests, batch_size)
        ) as executor:
            future_to_idx = {}
            for idx, tokens, payload in payloads:
                future = executor.submit(self._send_request, payload)
                future_to_idx[future] = (idx, tokens)

            for future in as_completed(future_to_idx):
                idx, tokens = future_to_idx[future]
                try:
                    resp_json = future.result()
                    results[idx] = (tokens, resp_json)
                except Exception as e:
                    raise RuntimeError(
                        f"Request {idx} failed: {e}"
                    ) from e

        # Parse all results
        responses = []
        teacher_topk_logprobs = []
        teacher_topk_indices = []

        for idx in range(batch_size):
            tokens, resp_json = results[idx]
            choice = resp_json["choices"][0]

            # Get generated token IDs
            gen_token_ids = choice.get("token_ids", [])
            if gen_token_ids is None:
                gen_token_ids = []

            # Build full response (prompt + generation)
            full_ids = list(tokens) + list(gen_token_ids)
            responses.append(torch.tensor(full_ids, dtype=torch.int32))

            if self.n_logprobs > 0 and choice.get("logprobs") is not None:
                logprobs_data = choice["logprobs"]

                # Parse generation (response) logprobs from top_logprobs
                response_lp, response_idx = self._parse_top_logprobs(
                    logprobs_data.get("top_logprobs", [])
                )

                if only_response:
                    teacher_topk_logprobs.append(response_lp)
                    teacher_topk_indices.append(response_idx)
                else:
                    # Parse prompt logprobs
                    prompt_lp_raw = choice.get("prompt_logprobs")
                    if prompt_lp_raw and len(prompt_lp_raw) > 1:
                        # prompt_logprobs[0] is always None (first token has no logprob)
                        # prompt_logprobs[i] is dict[int, {"logprob": float, "rank": int, ...}]
                        prompt_lp, prompt_idx = self._parse_prompt_logprobs(
                            prompt_lp_raw[1:]
                        )
                        teacher_topk_logprobs.append(
                            torch.vstack([prompt_lp, response_lp])
                        )
                        teacher_topk_indices.append(
                            torch.vstack([prompt_idx, response_idx])
                        )
                    else:
                        teacher_topk_logprobs.append(response_lp)
                        teacher_topk_indices.append(response_idx)

        return responses, teacher_topk_logprobs, teacher_topk_indices

    def _send_request(self, payload: dict) -> dict:
        """Send a single completion request to the vLLM server."""
        resp = self.session.post(
            f"{self.server_url}/v1/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _parse_top_logprobs(
        self, top_logprobs_list: List[Optional[dict]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse generation logprobs from OpenAI Completions format.

        top_logprobs_list: list of dicts, one per generated token.
        Each dict maps "token_id:XXX" -> logprob (float).
        The sampled token is the first entry in each dict.

        Returns:
            logprobs_tensor: [num_tokens, n_logprobs+1] float32
            indices_tensor: [num_tokens, n_logprobs+1] int32
        """
        n = self.n_logprobs + 1  # +1 because API returns n_logprobs + 1 (includes sampled)
        all_lp = []
        all_idx = []

        for step_dict in top_logprobs_list:
            if step_dict is None:
                # Shouldn't happen for generation tokens, but handle gracefully
                all_lp.append([0.0] * n)
                all_idx.append([0] * n)
                continue

            lps = []
            ids = []
            for tok_str, lp_val in step_dict.items():
                token_id = self._extract_token_id(tok_str)
                lps.append(lp_val)
                ids.append(token_id)
                if len(lps) >= n:
                    break

            # Pad if fewer than expected
            while len(lps) < n:
                lps.append(float("-inf"))
                ids.append(0)

            all_lp.append(lps[:n])
            all_idx.append(ids[:n])

        if not all_lp:
            # No tokens generated
            return (
                torch.zeros((0, n), dtype=torch.float32),
                torch.zeros((0, n), dtype=torch.int32),
            )

        return (
            torch.tensor(all_lp, dtype=torch.float32),
            torch.tensor(all_idx, dtype=torch.int32),
        )

    def _parse_prompt_logprobs(
        self, prompt_logprobs_list: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse prompt logprobs from vLLM's extended response format.

        prompt_logprobs_list: list of (dict[str_token_id, {logprob, rank, ...}] | None)
        Each dict maps token_id (as string int) -> Logprob info.

        Returns:
            logprobs_tensor: [num_tokens, n_logprobs+1] float32
            indices_tensor: [num_tokens, n_logprobs+1] int32
        """
        n = self.n_logprobs + 1
        all_lp = []
        all_idx = []

        for step in prompt_logprobs_list:
            if step is None:
                all_lp.append([0.0] * n)
                all_idx.append([0] * n)
                continue

            # step is dict: {token_id_str: {"logprob": float, "rank": int, ...}}
            # Sort by logprob descending to get top-k
            items = []
            for tid_str, info in step.items():
                tid = int(tid_str)
                lp = info["logprob"] if isinstance(info, dict) else float(info)
                items.append((tid, lp))

            # Sort by logprob descending
            items.sort(key=lambda x: x[1], reverse=True)

            lps = [x[1] for x in items[:n]]
            ids = [x[0] for x in items[:n]]

            # Pad
            while len(lps) < n:
                lps.append(float("-inf"))
                ids.append(0)

            all_lp.append(lps[:n])
            all_idx.append(ids[:n])

        if not all_lp:
            return (
                torch.zeros((0, n), dtype=torch.float32),
                torch.zeros((0, n), dtype=torch.int32),
            )

        return (
            torch.tensor(all_lp, dtype=torch.float32),
            torch.tensor(all_idx, dtype=torch.int32),
        )

    @staticmethod
    def _extract_token_id(tok_str: str) -> int:
        """Extract integer token ID from 'token_id:12345' format."""
        m = _TOKEN_ID_RE.match(tok_str)
        if m:
            return int(m.group(1))
        # Fallback: try direct int parse
        try:
            return int(tok_str)
        except (ValueError, TypeError):
            return 0

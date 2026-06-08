# Copyright 2025 Individual Contributor: furunding
# Adapted for vLLM v0.19 API compatibility.
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
vLLM Engine wrapper adapted for vLLM >= 0.19.

API changes from v0.12:
  - vllm.inputs.data.TokenInputs -> vllm.inputs.TokensPrompt
  - LogprobsTensors has 4 fields (added cu_num_generated_tokens)
  - _update_sample_logprobs receives LogprobsLists (numpy-based 4-tuple)
  - _update_prompt_logprobs receives LogprobsTensors (torch-based 4-tuple)
  - output.outputs[0].token_ids may be tuple instead of list
"""

import argparse
import random

import numpy as np
import torch
from codetiming import Timer
from transformers import AutoConfig
from vllm import LLM, SamplingParams

from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.inputs import TokensPrompt


# ============================================================================
# Monkey-patch LogprobsProcessor to store raw tensor/numpy logprobs
# instead of detokenizing into Logprob dicts.
# ============================================================================

def _update_prompt_logprobs(self, prompt_logprobs_tensors) -> None:
    """Store raw prompt logprobs tensors without detokenizing."""
    assert self.num_prompt_logprobs is not None
    assert self.prompt_logprobs is not None
    self.prompt_logprobs.append(prompt_logprobs_tensors)


def _update_sample_logprobs(self, logprobs_lists) -> None:
    """Store raw sample logprobs without building Logprob dicts."""
    assert self.num_logprobs is not None
    assert self.logprobs is not None
    assert self.cumulative_logprob is not None
    self.logprobs.append(logprobs_lists)


LogprobsProcessor._update_prompt_logprobs = _update_prompt_logprobs
LogprobsProcessor._update_sample_logprobs = _update_sample_logprobs


# ============================================================================
# VLLMEngine class
# ============================================================================

class VLLMEngine:
    def __init__(
        self,
        ckpt_path,
        n_logprobs=0,
        tp_size=1,
        pp_size=1,
        ep_size=1,
        gpu_memory_utilization=0.5,
        max_num_batched_tokens=8192,
        max_model_len=30720,
        distributed_executor_backend=None,
    ):
        self.n_logprobs = n_logprobs

        # Build extra kwargs for distributed / expert parallelism
        extra_kwargs = {}
        if distributed_executor_backend is not None:
            extra_kwargs["distributed_executor_backend"] = distributed_executor_backend
        if pp_size > 1:
            extra_kwargs["pipeline_parallel_size"] = pp_size
        if ep_size > 1:
            extra_kwargs["enable_expert_parallel"] = True

        self.llm = LLM(
            ckpt_path,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            max_logprobs=n_logprobs,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            # Limit parallel weight loading to avoid CPU OOM on large models
            max_parallel_loading_workers=2,
            **extra_kwargs,
        )

    def get_topk_logprobs(self, prompt_token_ids, temperature=0.8, max_new_tokens=1, only_response=False):
        def make_sampling_params(i=None):
            return SamplingParams(
                temperature=temperature,
                top_p=0.95,
                detokenize=False,
                logprobs=self.n_logprobs,
                prompt_logprobs=None if only_response else self.n_logprobs,
                max_tokens=max_new_tokens[i] if (i is not None) else max_new_tokens,
            )

        if isinstance(max_new_tokens, list):
            assert len(prompt_token_ids) == len(max_new_tokens)
            sampling_params = [make_sampling_params(i) for i in range(len(max_new_tokens))]
        else:
            sampling_params = make_sampling_params()

        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids[i]) for i in range(len(prompt_token_ids))]
        outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)

        responses, teacher_topk_logprobs, teacher_topk_indices = [], [], []
        for output in outputs:
            print('len(output.prompt_token_ids):', len(output.prompt_token_ids))
            print('len(output.outputs[0].token_ids):', len(output.outputs[0].token_ids))
            responses.append(torch.tensor(
                list(output.prompt_token_ids) + list(output.outputs[0].token_ids), dtype=torch.int32
            ))

            if self.n_logprobs > 0:
                # === Parse sample (response) logprobs ===
                # Monkey-patched _update_sample_logprobs stores a list of
                # LogprobsLists (numpy NamedTuples), one per decode step.
                sample_lp_list = output.outputs[0].logprobs
                if len(sample_lp_list) == 1:
                    response_topk_logprobs = torch.as_tensor(
                        np.array(sample_lp_list[0].logprobs), dtype=torch.float32
                    )
                    response_topk_indices = torch.as_tensor(
                        np.array(sample_lp_list[0].logprob_token_ids), dtype=torch.int32
                    )
                else:
                    all_lp = np.concatenate([lp.logprobs for lp in sample_lp_list], axis=0)
                    all_ids = np.concatenate([lp.logprob_token_ids for lp in sample_lp_list], axis=0)
                    response_topk_logprobs = torch.as_tensor(all_lp, dtype=torch.float32)
                    response_topk_indices = torch.as_tensor(all_ids, dtype=torch.int32)

                if only_response:
                    teacher_topk_logprobs.append(response_topk_logprobs)
                    teacher_topk_indices.append(response_topk_indices)
                else:
                    # === Parse prompt logprobs ===
                    # Monkey-patched _update_prompt_logprobs stores a list of
                    # LogprobsTensors (torch NamedTuples). Index [1] skips the
                    # first position (which has no logprob).
                    prompt_lp_list = output.prompt_logprobs
                    prompt_lp = prompt_lp_list[1]
                    prompt_topk_logprobs = prompt_lp.logprobs[:, :].to(torch.float32)
                    prompt_topk_indices = prompt_lp.logprob_token_ids[:, :].to(torch.int32)
                    teacher_topk_logprobs.append(torch.vstack([prompt_topk_logprobs, response_topk_logprobs]))
                    teacher_topk_indices.append(torch.vstack([prompt_topk_indices, response_topk_indices]))

        return responses, teacher_topk_logprobs, teacher_topk_indices


# ============================================================================
# Standalone test
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vLLM logprob (v0.19+)")
    parser.add_argument("model_dir", help="Model directory")
    parser.add_argument("--tp-size", type=int, default=1, help="TP size")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Test batch size")
    parser.add_argument("--seq-len", "-s", type=int, default=3840, help="Test sequence length")
    parser.add_argument("--token-file", "-t", type=str, help="Input token file")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_dir)
    print(f"Reading configs from {args.model_dir}: {config.vocab_size=}")

    prompt_token_ids = []
    if args.token_file:
        from get_batch import get_batch
        prompt_token_ids = get_batch()
    else:
        prompt_lens = args.batch_size * [args.seq_len]
        for pl in prompt_lens:
            prompt_token_ids.append([random.randint(1, config.vocab_size - 1000) for j in range(pl)])

    engine = VLLMEngine(ckpt_path=args.model_dir, n_logprobs=256, tp_size=args.tp_size)

    with Timer(name="get_topk_logprobs", initial_text=True):
        responses, teacher_topk_logprobs, teacher_topk_indices = engine.get_topk_logprobs(
            prompt_token_ids, temperature=0.7, max_new_tokens=1, only_response=True
        )

    print(f"responses[0].shape: {responses[0].shape}")
    if teacher_topk_logprobs:
        print(f"teacher_topk_logprobs[0].shape: {teacher_topk_logprobs[0].shape}")
        print(f"teacher_topk_indices[0].shape: {teacher_topk_indices[0].shape}")

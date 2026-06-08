# Copyright 2025 Individual Contributor: furunding
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

import argparse
import functools

import torch
import zmq
from codetiming import Timer
from utils import deserialize, serialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-addr", type=str, default="localhost:15556")
    parser.add_argument("--backend", type=str, default="vllm")
    parser.add_argument("--seq-len", type=int, default=3840)
    parser.add_argument("--n-logprobs", type=int, default=256)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--distributed-executor-backend", type=str, default=None,
                        help="Distributed executor backend (e.g. 'ray').")
    # vLLM engine knobs (forwarded to vllm_engine.VLLMEngine)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                        help="Fraction of GPU memory vLLM can use.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192,
                        help="vLLM max_num_batched_tokens.")
    parser.add_argument("--max-model-len", type=int, default=30720,
                        help="vLLM max_model_len (prompt+generation).")
    # vLLM HTTP engine options (backend=vllm_http)
    parser.add_argument("--vllm-server-url", type=str, default="http://localhost:8000",
                        help="URL of the vLLM OpenAI-compatible server (for vllm_http backend).")
    args = parser.parse_args()

    if args.backend == "vllm":
        from vllm_engine_v019 import VLLMEngine

        engine = VLLMEngine(
            args.ckpt_path,
            args.n_logprobs,
            args.tp_size,
            pp_size=args.pp_size,
            ep_size=args.ep_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            distributed_executor_backend=args.distributed_executor_backend,
        )
    elif args.backend == "vllm_http":
        from vllm_http_engine import VLLMHTTPEngine

        engine = VLLMHTTPEngine(
            server_url=args.vllm_server_url,
            model_name=args.ckpt_path,
            n_logprobs=args.n_logprobs,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}.")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # socket.bind(f"tcp://*:{port}")
    socket.connect(f"tcp://{args.proxy_addr}")

    print("worker started...", flush=True)

    # TODO: 新增prefix_cache_hit监控

    while True:
        message = socket.recv()
        try:
            with Timer(name="deserialize", initial_text=True, logger=functools.partial(print, flush=True)):
                request = deserialize(message)
        except Exception as e:
            print("[Server Error] Deserialize failed:", str(e), flush=True)
            socket.send(serialize({"status": "error", "reason": f"Deserialize failed: {e}"}))
            continue
        if isinstance(request, dict) and "prompt_token_ids" in request:
            prompt_token_ids = request["prompt_token_ids"]
            temperature = request.get("temperature", 1)
            max_tokens = request.get("max_tokens", 1)
            only_response = request.get("only_response", False)
            if isinstance(prompt_token_ids, torch.Tensor):
                prompt_token_ids = prompt_token_ids.tolist()
            with Timer(name="get_prompt_topk_logprobs", initial_text=True, logger=functools.partial(print, flush=True)):
                ### try and sendback error
                try:
                    responses, logps, indices = engine.get_topk_logprobs(
                        prompt_token_ids, temperature, max_new_tokens=max_tokens, only_response=only_response
                    )
                except Exception as e:
                    print("[Server Error] Exception occurred during generation:", str(e))
                    socket.send(serialize({"status": "error", "reason": f"Generate failed: {str(e)}"}))
                    continue
            with Timer(name="serialize", initial_text=True, logger=functools.partial(print, flush=True)):
                message = serialize(
                    {
                        "status": "ok",
                        "teacher_topk_logprobs": logps,
                        "teacher_topk_indices": indices,
                        "responses": responses,
                    }
                )
            with Timer(name="send", initial_text=True, logger=functools.partial(print, flush=True)):
                socket.send(message)

        else:
            socket.send(serialize({"status": "error", "reason": "invalid request format."}))


if __name__ == "__main__":
    main()

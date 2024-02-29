"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
# (prompt len, output len, [token latencies])
TOKEN_LATENCY: List[Tuple[int, int, List[float]]] = []
TIME_TO_FIRST_TOKEN: List[float] = []
TEMPERATURE = 0.0

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    config_input_len: int,
    config_output_len: int,
) -> List[Tuple[str, int, int]]:
    return [("hi " * config_input_len, config_input_len, config_output_len)]
async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> None:

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else TEMPERATURE,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": True,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    request_start_time = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                token_latencies = []
                previous_token_time = time.perf_counter()
                first = True
                async for chunk, _ in response.content.iter_chunks():
                    # Stream on: Each chunk in the response is the full response so far
                    chunks = [chunk]

                    now_time = time.perf_counter()
                    if first:
                        time_to_first = now_time - previous_token_time
                        first = False
                    else:
                        token_latencies.append(now_time - previous_token_time)
                    previous_token_time = now_time

                    # Stream off: Chunks are full response.
                    # chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = output[:-1]  # Get rid of EOF
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    TOKEN_LATENCY.append((prompt_len, output_len, token_latencies))
    TIME_TO_FIRST_TOKEN.append(time_to_first)

async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []

    # exp_start_time = time.time()
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(send_request(backend, api_url, prompt,
                                                prompt_len, output_len,
                                                best_of, use_beam_search))
        tasks.append(task)

        # Run the experiment for 2 minutes (or until all prompts are served).
        # if time.time() - exp_start_time > 120:
            # break
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer, args.input_len, args.output_len)

    benchmark_start_time = time.perf_counter()
    asyncio.run(benchmark(args.backend, api_url, input_requests, args.best_of,
                          args.use_beam_search, args.request_rate))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print()
    print("RESULT SUMMARY")
    print(f"Request rate: {args.request_rate} req/s")
    print(f"Prompt count: {len(REQUEST_LATENCY)}")
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Request Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")
    print(f"Output Token Throughput: {sum([output for _, output, _ in REQUEST_LATENCY]) / benchmark_time:.2f} tokens/s")
    print()

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print("REQUEST LATENCIES")
    print(f"Avg: {avg_latency:.2f} s")
    print(f"50p: {np.percentile([latency for _, _, latency in REQUEST_LATENCY], 50)} s")
    print(f"90p: {np.percentile([latency for _, _, latency in REQUEST_LATENCY], 90)} s")
    print(f"99p: {np.percentile([latency for _, _, latency in REQUEST_LATENCY], 99)} s")
    print()

    print()

    all_token_latencies = np.array([token_latencies for _, _, token_latencies in TOKEN_LATENCY])
    print("TOKEN LATENCIES")
    print("TTFT")
    print(f'Avg: {np.mean(TIME_TO_FIRST_TOKEN)}')
    print(f'50p: {np.percentile(TIME_TO_FIRST_TOKEN, 50)}')
    print(f'90p: {np.percentile(TIME_TO_FIRST_TOKEN, 90)}')
    print(f'99p: {np.percentile(TIME_TO_FIRST_TOKEN, 99)}')
    print("TPOT")
    print(f'Avg: {np.mean(all_token_latencies)}')
    print(f'50p: {np.percentile(all_token_latencies, 50)}')
    print(f'90p: {np.percentile(all_token_latencies, 90)}')
    print(f'99p: {np.percentile(all_token_latencies, 99)}')
    print()

    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    print(f"Token latency 50p: {np.percentile([latency / (prompt_len + output_len) for prompt_len, output_len, latency in REQUEST_LATENCY], 50)} s")
    print(f"Token latency 90p: {np.percentile([latency / (prompt_len + output_len) for prompt_len, output_len, latency in REQUEST_LATENCY], 90)} s")
    print(f"Token latency 99p: {np.percentile([latency / (prompt_len + output_len) for prompt_len, output_len, latency in REQUEST_LATENCY], 99)} s")
    avg_per_output_token_latency = np.mean([
        latency / output_len
        for _, output_len, latency in REQUEST_LATENCY
    ])
    print("MISLEADING: Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")
    print()

    print("SEQUENCE LENGTHS")
    print(f"Sequence average: {np.mean([input+output for input, output, _ in REQUEST_LATENCY])} tokens")
    print(f"Sequence length 50p: {np.percentile([input+output for input, output, _ in REQUEST_LATENCY], 50)} tokens")
    print(f"Sequence length 90p: {np.percentile([input+output for input, output, _ in REQUEST_LATENCY], 90)} tokens")
    print(f"Sequence length 99p: {np.percentile([input+output for input, output, _ in REQUEST_LATENCY], 99)} tokens")
    print()
    print('Easy Paste:')
    # inlen,outlen,reqrate,numPrompts,totalTime,reqThroughput, outputTokenThroughput, 50p req lat, 99p req lat, 50p TTFT, 99p TTFT
    print(f'=SPLIT("{args.input_len},{args.output_len},{args.request_rate},{len(REQUEST_LATENCY)},{benchmark_time:.2f},{len(REQUEST_LATENCY) / benchmark_time},{sum([output for _, output, _ in REQUEST_LATENCY]) / benchmark_time},{np.percentile([latency for _, _, latency in REQUEST_LATENCY], 50)},{np.percentile([latency for _, _, latency in REQUEST_LATENCY], 90)},{np.percentile([latency for _, _, latency in REQUEST_LATENCY], 99)},{np.percentile(TIME_TO_FIRST_TOKEN, 50)},{np.percentile(TIME_TO_FIRST_TOKEN, 90)},{np.percentile(TIME_TO_FIRST_TOKEN, 99)}", ",")')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "tgi"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument("--input_len", type=int, default=0)
    parser.add_argument("--output_len", type=int, default=0)
    args = parser.parse_args()
    main(args)
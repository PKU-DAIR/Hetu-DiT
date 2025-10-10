import argparse
import asyncio
from hetu_dit.logger import init_logger
import os
import time
from typing import List, Optional

import ast
from aiohttp import ClientSession, ClientTimeout


os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

logger = init_logger(__name__)


class Request:
    def __init__(
        self,
        req_id,
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
        num_inference_steps,
        req_time,
    ):
        self.req_id = req_id
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.req_time = req_time

    def __repr__(self):
        return (
            f"req_id={self.req_id}, prompt={self.prompt}, "
            f"negative_prompt={self.negative_prompt}, "
            f"height={self.height}, width={self.width}, num_frames={self.num_frames} "
            f"num_inference_steps={self.num_inference_steps}, "
            f"req_time={self.req_time}"
        )


def _literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.List):
        return [_literal(x) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(x) for x in node.elts)
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    raise ValueError(f"Unsupported AST literal: {ast.dump(node)}")


def parse_request_line(line: str) -> Optional[dict]:
    text = line.strip()
    if not text or not text.startswith("Request(") or not text.endswith(")"):
        return None
    try:
        node = ast.parse(text, mode="eval").body
        if not isinstance(node, ast.Call):
            return None
        out = {}
        for kw in node.keywords:
            out[kw.arg] = _literal(kw.value)
        return out
    except Exception:
        return None


def load_requests_from_trace(trace_path: str) -> List[Request]:
    reqs: List[Request] = []
    with open(trace_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = parse_request_line(line)
            if not rec:
                continue

            # Prefer to use request_id_local
            req_id_local = rec.get("request_id_local")
            request_id = rec.get("request_id")
            req_id = str(
                req_id_local
                if req_id_local is not None
                else request_id
                if request_id is not None
                else 0
            )

            # Required fields
            try:
                timestamp = float(rec["timestamp"])
                height = int(rec["height"])
                width = int(rec["width"])
                num_frames = int(rec["num_frames"])
                steps = int(rec["num_inference_steps"])
            except Exception:
                continue  # Skip incomplete lines

            prompt = str(rec.get("prompt", ""))
            negative_prompt = str(rec.get("negative_prompt", ""))

            req = Request(
                req_id=req_id,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=steps,
                req_time=max(0.0, timestamp),
            )
            reqs.append(req)

    # Sort by time to ensure replay order
    reqs.sort(key=lambda r: r.req_time)
    return reqs


async def send_request(
    server: str,
    req_id: str,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    session: ClientSession,
) -> None:
    request_start_time = time.time()
    headers = {"Content-Type": "application/json", "User-Agent": "Batch Client"}
    url = server + "/generate"

    data = {
        "req_id": req_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
    }

    try:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                logger.info(f"Request {req_id} completed successfully")
            elif response.status == 202:
                logger.info(f"Request {req_id} accepted.")
            else:
                logger.error(f"Request {req_id} failed with status {response.status}")
    except Exception as e:
        logger.error(f"Request {req_id} raised error: {e}")

    request_end_time = time.time()
    logger.info(
        f"req_id {req_id} height {height} width {width} num_frames {num_frames} "
        f"latency {request_end_time - request_start_time:.2f} s"
    )


async def delayed_send(req: Request, server: str, session: ClientSession):
    await asyncio.sleep(req.req_time)
    await send_request(
        server,
        req.req_id,
        req.prompt,
        req.negative_prompt,
        req.height,
        req.width,
        req.num_frames,
        req.num_inference_steps,
        session,
    )


async def benchmark(server: str, requests: List[Request]) -> None:
    timeout = ClientTimeout(total=3600)
    async with ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(delayed_send(req, server, session)) for req in requests
        ]
        await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(
        description="Batch HetuDiT API requests (replay from trace)."
    )
    parser.add_argument("--url", type=str, required=True, help="API server URL")

    parser.add_argument(
        "--trace", type=str, required=True, help="Path to converted_trace_*.txt"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful landscape",
        help="(ignored when --trace is used)",
    )
    parser.add_argument(
        "--negative-prompt", type=str, default="", help="(ignored when --trace is used)"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="(ignored when --trace is used)"
    )
    parser.add_argument(
        "--num-requests", type=int, default=10, help="(ignored when --trace is used)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.1, help="(ignored when --trace is used)"
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default="sd3",
        help="(ignored when --trace is used) sd3 or cogvideox",
    )

    args = parser.parse_args()

    requests = load_requests_from_trace(args.trace)
    if not requests:
        logger.error(f"no valid requests parsed from trace: {args.trace}")
        return

    logger.info(
        f"Loaded {len(requests)} requests from {args.trace}. "
        f"time range: {requests[0].req_time:.3f} ~ {requests[-1].req_time:.3f}s"
    )

    asyncio.run(benchmark(args.url, requests))


if __name__ == "__main__":
    main()

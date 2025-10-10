import argparse
import asyncio
from hetu_dit.logger import init_logger
import random
import time
from typing import List
from aiohttp import ClientSession, ClientTimeout
import os

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

    async with session.post(url, headers=headers, json=data) as response:
        if response.status == 200:
            logger.info(f"Request {req_id} completed successfully")
        elif response.status == 202:
            logger.info(f"Request {req_id} accepted.")
        else:
            logger.error(f"Request {req_id} failed with status {response.status}")

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
    parser = argparse.ArgumentParser(description="Batch HetuDiT API requests")
    parser.add_argument("--url", type=str, required=True, help="API server URL")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful landscape",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt to avoid certain features",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of inference steps"
    )
    parser.add_argument(
        "--num-requests", type=int, default=10, help="Number of requests to send"
    )
    parser.add_argument(
        "--delay", type=float, default=0.1, help="Delay between each request in seconds"
    )

    parser.add_argument(
        "--model-class", type=str, default="sd3", help="sd3 or cogvideox"
    )

    args = parser.parse_args()
    random.seed(42)
    if args.model_class == "sd3" or args.model_class == "sd3.5":
        size_options = [256, 512, 768, 1024, 1280, 1536]
        num_frames_options = [1]
    elif args.model_class == "flux":
        size_options = [128, 256, 512, 1024, 2048, 3072, 4096]
        num_frames_options = [1]
    elif args.model_class == "hunyuanvideo":
        size_options = [384, 768, 1024]
        num_frames_options = [5, 9, 33, 65]
    elif args.model_class == "hunyuandit":
        size_options = [768, 1024, 1280, 1536]
        num_frames_options = [1]
    elif args.model_class == "cogvideox":
        size_options = [384, 768, 1024]
        num_frames_options = [1, 5, 9, 33, 65, 81]

    requests = []

    for i in range(0, args.num_requests):
        height = random.choice(size_options)
        width = random.choice(size_options)
        num_frames = random.choice(num_frames_options)

        req = Request(
            req_id=str(i),
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=height,
            num_frames=num_frames,
            num_inference_steps=args.steps,
            req_time=i * args.delay,
        )
        requests.append(req)
    asyncio.run(benchmark(args.url, requests))


if __name__ == "__main__":
    main()

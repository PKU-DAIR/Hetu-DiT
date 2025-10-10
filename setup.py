from setuptools import find_packages, setup
import subprocess


def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = [line for line in nvcc_version.split("\n") if "release" in line][
            0
        ]
        cuda_version = version_line.split(" ")[-2].replace(",", "")
        return "cu" + cuda_version.replace(".", "")
    except Exception as e:
        return "no_cuda"


if __name__ == "__main__":
    fp = open("hetu_dit/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="hetu_dit",
        packages=find_packages(),
        install_requires=[
            "torch>=2.1.0",
            "accelerate>=0.33.0",
            "diffusers>=0.31",  # NOTE: diffusers>=0.31.0 is necessary for Flux
            "transformers>=4.39.1",
            "sentencepiece>=0.1.99",
            "beautifulsoup4>=4.12.3",
            # "distvae",
            "yunchang==0.3.5",
            "pytest",
            "flask",
            "opencv-python",
            "ray==2.39.0",
            "fastapi==0.110.0",
            "uvicorn==0.28.0",
            "aiohttp",
            "pulp",
            "nixl"
        ],
        extras_require={
            "flash_attn": [
                "flash_attn>=2.6.3",
            ],
        },
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )

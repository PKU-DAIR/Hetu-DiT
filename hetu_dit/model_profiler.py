import os
import json
import numpy as np
from typing import Dict
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from hetu_dit.logger import init_logger

logger = init_logger(__name__)


class ModelProfiler:
    def __init__(self, model_name: str, device: str, cache_dir: str = None):
        self.model_name = model_name
        self.device = device
        if cache_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(current_dir, "profile_cache")
        self.cache_dir = os.path.abspath(cache_dir)

        self.profile_data = defaultdict(list)
        self.average_data = defaultdict(list)
        self.models = defaultdict(dict)
        self.dict_models = defaultdict(dict)
        self.steps = 10  # diffusion steps for profiling
        self.model_data = {
            "cogvideox": {
                "768-1360-81": {1: 30.5, 2: 17.75, 4: 9.1, 8: 4.75},  # 81600
                "768-1360-33": {1: 9.37, 2: 5.63, 4: 3, 8: 1.57},  # 32640
                "768-1360-65": {1: 22.14, 2: 13.07, 4: 6.77, 8: 3.56},  # 65280
                "768-432-81": {1: 5.72, 2: 3.51, 4: 1.89, 8: 1.01},  # 25920
                "768-432-65": {1: 4.42, 2: 2.76, 4: 1.49, 8: 0.826446281},  # 20736
                "768-432-33": {
                    1: 2.16,
                    2: 0.751879699,
                    4: 0.751879699,
                    8: 0.452488688,
                },  # 10368
                "768-432-17": {1: 1.19, 2: 0.763358779, 4: 0.45045045, 8: 0.3},  # 5184
                "768-432-1": {
                    1: 0.384615385,
                    2: 0.262467192,
                    4: 0.225733634,
                    8: 0.158730159,
                },  # 1296
                "768-1360-1": {
                    1: 1.26,
                    2: 0.78125,
                    4: 0.483091787,
                    8: 0.297619048,
                },  # 4080
                "768-1360-17": {1: 4.71, 2: 2.91, 4: 1.55, 8: 0.840336134},  # 16320
            },
            "hunyuanvideo": {
                "720-1280-129": {1: 118, 2: 64.48, 4: 31.92, 8: 16.35},  # 115200
                "720-1280-65": {1: 37.53, 2: 21, 4: 10.7, 8: 5.8},  # 57600
                "720-1280-33": {1: 13.7, 2: 8.01, 4: 4.1, 8: 2.35},  # 28800
                "720-1280-17": {1: 5.84, 2: 3.61, 4: 1.86, 8: 1.08},  # 14400
                "720-1280-1": {1: 0.787401575, 2: 0.68, 4: 0.48, 8: 0.38},  # 3600
                "960-544-129": {1: 43.58, 2: 24.6, 4: 12.4, 8: 6.67},  # 65280
                "960-544-65": {1: 15.09, 2: 8.76, 4: 4.51, 8: 2.3},  # 32640
                "960-544-33": {1: 6.01, 2: 3.68, 4: 2.03, 8: 1},  # 16320
                "960-544-17": {1: 2.77, 2: 1.83, 4: 1.02, 8: 0.68},  # 8160
                "960-544-1": {1: 0.54, 2: 0.46, 4: 0.38, 8: 0.3},  # 2040
            },
            "flux": {
                "128-128-1": {
                    1: 0.054288817,
                    2: 0.055005501,
                    4: 0.050556117,
                    8: 0.046296296,
                },  # 64
                "256-256-1": {
                    1: 0.108577633,
                    2: 0.110011001,
                    4: 0.101112235,
                    8: 0.092592593,
                },  # 256
                "512-512-1": {
                    1: 0.22172949,
                    2: 0.194931774,
                    4: 0.136612022,
                    8: 0.121212121,
                },  # 1024
                "1024-1024-1": {
                    1: 0.840336134,
                    2: 0.537634409,
                    4: 0.323624595,
                    8: 0.227272727,
                },  # 4096
                "2048-2048-1": {1: 4.67, 2: 2.85, 4: 1.48, 8: 1.28},  # 16384
                "3072-3072-1": {1: 15.5, 2: 9, 4: 4.55, 8: 2.46},  # 36864
                "4096-4096-1": {1: 39.83, 2: 22.97, 4: 11.76, 8: 5.66},  # 65536
            },
            "sd3": {
                "128-128-1": {
                    1: 0.022045855,
                    2: 0.022045855,
                    4: 0.022045855,
                    8: 0.022045855,
                },  # 64
                "256-256-1": {
                    1: 0.029664788,
                    2: 0.029664788,
                    4: 0.029664788,
                    8: 0.029664788,
                },  # 256
                "512-512-1": {
                    1: 0.059171598,
                    2: 0.063011972,
                    4: 0.063011972,
                    8: 0.063011972,
                },  # 1024
                "1024-1024-1": {
                    1: 0.242718447,
                    2: 0.178571429,
                    4: 0.166666667,
                    8: 0.153846154,
                },  # 4096
                "1536-1536-1": {
                    1: 0.595238095,
                    2: 0.367430923,
                    4: 0.297619048,
                    8: 0.17303433,
                },  # 9216
            },
        }
        self.best_batchsize = {
            "sd3": {16384: 4, 65536: 2, 131072: 2, 262144: 1, 524288: 1, 1048576: 1},
            "flux": {16384: 4, 65536: 2, 131072: 2, 262144: 1, 524288: 1, 1048576: 1},
        }

    def generate_task_config(self):
        """Generate a list of task configurations to profile."""
        parallel_values = [1, 2, 4, 8]
        world_degree = 8
        parallel_configs = []
        max_bs = 1
        if self.model_name == "sd3":
            max_bs = 8
            frame = [1]
            image_sizes = [128, 256, 512, 1024]
            for ulysses, ring, tp, pipefusion in product(parallel_values, repeat=4):
                if ulysses * ring * tp * pipefusion <= world_degree:
                    parallel_configs.append((ulysses, ring, tp, pipefusion))
        elif "flux" in self.model_name:
            max_bs = 4
            parallel_values = [2]
            world_degree = 2
            frame = [1]
            image_sizes = [1024]
            parallel_configs.append((1, 1, 1, 2))

        elif "hunyuanvideo" in self.model_name:
            frame = [1, 9, 33, 65]
            image_sizes = [384, 768]
            for ulysses, ring, tp in product(parallel_values, repeat=3):
                if ulysses * ring * tp <= world_degree:
                    parallel_configs.append((ulysses, ring, tp, 1))
        elif "cogvideox" in self.model_name:
            frame = [1, 9, 33, 65, 81]
            image_sizes = [384, 768]
            for ulysses, ring, tp in product(parallel_values, repeat=3):
                if ulysses * ring * tp <= world_degree:
                    parallel_configs.append((ulysses, ring, tp, 1))

        cases = [
            (parallel_config, height, width, frame)
            for parallel_config, height, width, frame in product(
                parallel_configs, image_sizes, image_sizes, frame
            )
            if height >= width
        ]
        return cases, max_bs

    def save_data(self, tag, results: list):
        keys = []
        for stage in ["encode", "diffusion", "vae"]:
            keys.append(stage)
            keys.append(f"{stage}_mem")
        for result in results:
            re = {
                key: result["inner_results"][key]
                for key in keys
                if key in result["inner_results"]
            }
            self.profile_data[tag].append(re)

    def parse_results(self):
        """Parse the raw profiling results to compute averages."""
        averages = {}
        for key, results in self.profile_data.items():
            parts = key.split("_")
            height = int(parts[3])
            width = int(parts[5])
            frame = int(parts[7])
            batchsize = int(parts[9]) if len(parts) > 9 else 1
            parallel_config = parts[1]

            stages = ["encode", "diffusion", "vae"]
            times = {stage: [] for stage in stages}
            mems = {stage: [] for stage in stages}

            for result in results:
                for stage in stages:
                    if stage in result:
                        times[stage].append(result[stage])
                    if f"{stage}_mem" in result:
                        mems[stage].append(result[f"{stage}_mem"])
            averages[key] = {
                "height": height,
                "width": width,
                "frame": frame,
                "batchsize": batchsize,
                "parallel_config": parallel_config,
                **{
                    f"{stage}_avg_time": np.mean(times[stage]) if times[stage] else None
                    for stage in stages
                },
                **{
                    f"{stage}_avg_mem": np.mean(mems[stage]) if mems[stage] else None
                    for stage in stages
                },
            }

        for key, config in averages.items():
            parallel = config["parallel_config"]
            self.average_data[parallel].append(config)

    def process_profile_data(self):
        """
        Processes the raw profile data and transforms it into the desired format.
        """

        processed_data = defaultdict(lambda: defaultdict(lambda: float("inf")))

        for parallel_config_str, results in self.average_data.items():
            parallel_config = tuple(
                map(int, parallel_config_str.strip("()").split(","))
            )
            # Calculate the total parallel degree
            total_degree = int(np.prod(parallel_config))

            for result in results:
                h, w, f = result["height"], result["width"], result["frame"]
                time = result["diffusion_avg_time"] / self.steps

                # Create the outer key
                key = self._format_key(h, w, f)

                # Update with the minimum time for the given key and total_degree
                if time < processed_data[key][total_degree]:
                    processed_data[key][total_degree] = time

        self.model_data = {self.model_name: processed_data}

        return self.model_data

    def process_profile_cache(self):
        """Process the average_data to compute best batch sizes and save to a JSON file."""
        data = self.average_data["(1, 1, 1, 1)"]
        group = defaultdict(list)
        for item in data:
            h = item.get("height")
            w = item.get("width")
            f_ = item.get("frame", 1)
            bs = item.get("batchsize", 1)
            seqlen = h * w * f_ if h and w else None
            diffusion = item.get("diffusion_avg_time")
            if seqlen and bs and diffusion is not None:
                group[(seqlen, bs)].append(diffusion)
        nested = {}
        sorted_keys = sorted(group.keys(), key=lambda x: (x[0], x[1]))
        for seqlen, bs in sorted_keys:
            if seqlen not in nested:
                nested[seqlen] = {}
            nested[seqlen][bs] = float(np.mean(group[(seqlen, bs)]))
        ordered_nested = {}
        for seqlen in sorted(nested.keys()):
            ordered_nested[seqlen] = {}
            for bs in sorted(nested[seqlen].keys()):
                ordered_nested[seqlen][bs] = nested[seqlen][bs]
        result = ordered_nested
        cache_file = os.path.join(
            self.cache_dir, f"{self.model_name}_{self.device}_batch.json"
        )
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=4)

    def compute_all_best_batchsizes(self, prop=0.2, batch_json_path=None):
        """
        Compute the best batch sizes for all sequence lengths and save to a JSON file.

        Args:
            prop (float): Proportional threshold to determine the best batch size.
        """
        if batch_json_path is None:
            batch_json_path = os.path.join(
                self.cache_dir, f"{self.model_name}_{self.device}_batch.json"
            )
        if not os.path.exists(batch_json_path):
            self.process_profile_cache()
        with open(batch_json_path, "r") as f:
            batch_data = json.load(f)
        self.best_batchsize[self.model_name] = {}
        for seqlen, batchsize_dict in batch_data.items():
            sorted_bs = sorted(
                [
                    int(bs)
                    for bs in batchsize_dict.keys()
                    if int(bs) > 0 and (int(bs) & (int(bs) - 1)) == 0
                ]
            )
            best_bs = sorted_bs[0]
            prev_time = batchsize_dict[str(best_bs)]
            for bs in sorted_bs[1:]:
                cur_time = batchsize_dict[str(bs)]
                if cur_time <= prev_time * (1 + prop):
                    best_bs = bs
                    prev_time = cur_time
                else:
                    break
            self.best_batchsize[self.model_name][int(seqlen)] = int(best_bs)

    def query_best_batchsize(self, height, width, frame: int = 1):
        # only use batch for non-parallel
        seqlen = height * width * frame
        if self.model_name not in self.best_batchsize:
            return 1
        seqlen_list = list(self.best_batchsize[self.model_name].keys())
        closest_seqlen = min(seqlen_list, key=lambda x: abs(x - seqlen))
        return self.best_batchsize[self.model_name][closest_seqlen]

    def _fit_curves_from_model_data(self):
        """
        Fits curves using the pre-loaded self.model_data.
        """
        # If models are already fitted, skip.
        if self.dict_models:
            return

        data_for_fitting = defaultdict(list)
        if not self.model_data.get(self.model_name):
            raise ValueError(
                f"No model data found for {self.model_name} to fit curves."
            )

        model_specific_data = self.model_data[self.model_name]
        for h_w_f, perf_dict in model_specific_data.items():
            h, w, f = map(int, h_w_f.split("-"))
            for degree, time in perf_dict.items():
                # Group data by parallel degree for fitting
                data_for_fitting[str(degree)].append(
                    {
                        "height": h,
                        "width": w,
                        "frame": f,
                        "diffusion_avg_time": time,
                    }
                )

        for degree, configs in data_for_fitting.items():
            # We only have diffusion time, so we only fit for that stage.
            stage = "diffusion"

            valid_data = [
                (c["height"] * c["width"] * c["frame"], c[f"{stage}_avg_time"])
                for c in configs
                if c.get(f"{stage}_avg_time") is not None
            ]

            # Need at least 3 points for a quadratic fit
            if len(valid_data) < 3:
                print(
                    f"Warning: Not enough data points ({len(valid_data)}) to fit curve for degree {degree}. Skipping."
                )
                continue

            valid_X, y_time = zip(*valid_data)

            try:
                model_time, _ = curve_fit(self.quadratic_func, valid_X, y_time)

                if degree not in self.dict_models:
                    self.models[degree] = {}

                self.dict_models[degree][stage] = {
                    "time_model": model_time,
                    "mem_model": [0, 0],  # No memory data available from this source
                }
            except RuntimeError as e:
                print(f"Error fitting curve for degree {degree}: {e}")

    def get_performance_data(self, model_name, key) -> Dict[int, float]:
        """
        Gets the performance dictionary for a given model and size.
        If an exact match for the size exists in the profile data, it's returned.
        Otherwise, it uses the fitted curves to predict the performance.
        """
        height, width, frame = map(int, key.split("-"))
        if model_name not in self.model_data:
            return {1: 8, 2: 4, 4: 2, 8: 1}  # Default fallback

        model_specific_data = self.model_data[model_name]

        # 1. Check for an exact match in the existing data
        reverse_key = f"{width}-{height}-{frame}"
        if key in model_specific_data:
            # print(f"Found exact match for {key}")
            return model_specific_data[key]
        elif reverse_key in model_specific_data:
            # print(f"Found exact match for {reverse_key}")
            return model_specific_data[reverse_key]

        # 2. If no match, predict using fitted curves from model_data
        # print(f"No exact match for {key}. Predicting from fitted curves.")
        if not self.dict_models:
            self._fit_curves_from_model_data()

        if not self.dict_models:
            raise RuntimeError(
                "Could not fit any curves. Not enough data in model_data."
            )

        predicted_times = {}
        X = height * width * frame

        for degree_str, stage_models in self.dict_models.items():
            if (
                "diffusion" in stage_models
                and "time_model" in stage_models["diffusion"]
            ):
                model_time = stage_models["diffusion"]["time_model"]
                predicted_time = self.quadratic_func(X, *model_time)
                degree = int(degree_str)
                predicted_times[degree] = predicted_time

        if not predicted_times:
            raise RuntimeError(
                "Prediction failed. No valid diffusion models were fitted."
            )

        return dict(sorted(predicted_times.items()))

    def load_cache(self):
        cache_file = os.path.join(
            self.cache_dir, f"{self.model_name}_{self.device}.json"
        )
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                print(f"Loading profile data from {cache_file}")
                self.average_data = json.load(f)
        cache_file_dict = os.path.join(
            self.cache_dir, f"{self.model_name}_{self.device}_dict.json"
        )
        if os.path.exists(cache_file_dict):
            with open(cache_file_dict, "r") as f:
                print(f"Loading model data from {cache_file_dict}")
                self.model_data = json.load(f)
                # Convert string keys back to integers
                for model, configs in self.model_data.items():
                    for shape, inner in configs.items():
                        new_inner = {int(k): v for k, v in inner.items()}
                        configs[shape] = new_inner
                return True
        return False

    def save_cache(self):
        self.parse_results()
        os.makedirs(self.cache_dir, exist_ok=True)

        cache_file = os.path.join(
            self.cache_dir, f"{self.model_name}_{self.device}.json"
        )
        with open(cache_file, "w") as f:
            json.dump(self.average_data, f, indent=4)

        cache_file_processed = os.path.join(
            self.cache_dir, f"{self.model_name}_{self.device}_dict.json"
        )
        processed_data = self.process_profile_data()
        with open(cache_file_processed, "w") as f:
            json.dump(processed_data, f, indent=4)

        self.compute_all_best_batchsizes()

    def quadratic_func(self, x, a, b, c):
        return a * x**2 + b * x + c

    def linear_func(self, x, a, b):
        return a * x + b

    def fit_curves(self):
        """Fit curves for and for memory."""
        for parallel, configs in self.average_data.items():
            X = [c["height"] * c["width"] * c["frame"] for c in configs]
            stage_models = {}
            for stage in ["encode", "diffusion", "vae"]:
                valid_data = [
                    (x, c[f"{stage}_avg_time"], c[f"{stage}_avg_mem"] / 1024**3)
                    for x, c in zip(X, configs)
                    if c[f"{stage}_avg_time"] is not None
                    and c[f"{stage}_avg_mem"] is not None
                ]

                valid_X, y_time, y_mem = zip(*valid_data)

                model_time, _ = curve_fit(self.quadratic_func, valid_X, y_time)

                model_mem, _ = curve_fit(self.linear_func, valid_X, y_mem)

                stage_models[stage] = {"time_model": model_time, "mem_model": model_mem}
            self.models[parallel] = stage_models

    def select_best_strategy(
        self, height, width, frame: int = 1, max_degree: int = 8, mem_limit=None
    ):
        if not self.models:
            self.fit_curves()

        diffusion_time = {}
        for parallel, stage_models in self.models.items():
            X = height * width * frame
            model_time = stage_models["diffusion"]["time_model"]
            diffusion_time[parallel] = self.quadratic_func(X, *model_time)

        best_strategy = None
        min_time = float("inf")

        for parallel, time in diffusion_time.items():
            parallel_config = tuple(map(int, parallel.strip("()").split(",")))
            if mem_limit:
                model_mem = stage_models["diffusion"]["mem_model"]
                predicted_mem = self.linear_func(height * width, *model_mem)
                if predicted_mem > mem_limit:
                    continue

            if np.prod(parallel_config) > max_degree:
                continue

            if time < min_time:
                min_time = time
                best_strategy = parallel_config

        return best_strategy, {"diffusion_time": min_time}

    def paint(self, tag="(2, 1, 1, 1)"):  # useless
        data = self.average_data[tag]
        areas = []
        times = []
        mems = []

        for entry in data:
            area = entry["height"] * entry["width"] * entry["frame"]
            areas.append(area)
            times.append(entry["diffusion_avg_time"])
            mems.append(entry["diffusion_avg_mem"] / 1024**3)

        areas = np.array(areas)
        times = np.array(times)
        mems = np.array(mems)

        if not self.models:
            self.fit_curves()

        residuals_time_quad = times - self.quadratic_func(
            areas, *self.models[tag]["diffusion"]["time_model"]
        )
        ss_res_time_quad = np.sum(residuals_time_quad**2)
        ss_tot_time_quad = np.sum((times - np.mean(times)) ** 2)
        r_squared_time_quad = 1 - (ss_res_time_quad / ss_tot_time_quad)

        # Calculate R-squared for both fits
        residuals_mem_lin = mems - self.linear_func(
            areas, *self.models[tag]["diffusion"]["mem_model"]
        )
        ss_res_mem_lin = np.sum(residuals_mem_lin**2)
        ss_tot_mem_lin = np.sum((mems - np.mean(mems)) ** 2)
        r_squared_mem_lin = 1 - (ss_res_mem_lin / ss_tot_mem_lin)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Time vs Area plot
        x_fit = np.linspace(min(areas), max(areas), 100)
        ax1.scatter(areas, times, label="Actual Data")
        ax1.plot(
            x_fit,
            self.quadratic_func(x_fit, *self.models[tag]["diffusion"]["time_model"]),
            label=f"Quadratic Fit (R²={r_squared_time_quad:.3f})",
        )
        ax1.set_xlabel("Image Area (height × width)")
        ax1.set_ylabel("Diffusion Time (s)")
        ax1.set_title("Diffusion Time vs Image Area")
        ax1.legend()
        ax1.grid(True)

        # Memory vs Area plot
        ax2.scatter(areas, mems, label="Actual Data")
        ax2.plot(
            x_fit,
            self.linear_func(x_fit, *self.models[tag]["diffusion"]["mem_model"]),
            label=f"Linear Fit (R²={r_squared_mem_lin:.3f})",
        )
        ax2.set_xlabel("Image Area (height × width)")
        ax2.set_ylabel("Diffusion Memory (GB)")
        ax2.set_title("Diffusion Memory vs Image Area")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"fit_image/diffusion_{tag}.png")
        plt.close()

    def _format_key(self, height, width, frame):
        return f"{height}-{width}-{frame}"

    def get_running_time(
        self, height, width, frame: int = 1, degree: int = 1, model: str = "cogvideox"
    ) -> float:
        key = self._format_key(height, width, frame)
        if model not in self.model_data or key not in self.model_data[model]:
            raise ValueError("Unsupported configuration")
        if degree not in self.model_data[model][key]:
            raise ValueError("Unsupported degree for this configuration")
        return self.model_data[model][key][degree]

    def select_best_strategy_fixed(
        self,
        height,
        width,
        frame: int = 1,
        max_degree: int = 8,
        model: str = "cogvideox",
    ) -> int:
        key = self._format_key(height, width, frame)
        if model not in self.model_data or key not in self.model_data[model]:
            raise ValueError("Unsupported configuration")

        data = self.model_data[model][key]
        base_time = data[1]  # running time with degree 1
        best_degree = 1

        for degree in sorted(data.keys()):
            if degree > max_degree or degree == 1:
                continue
            actual_speedup = base_time / data[degree]
            theoretical_speedup = degree
            efficiency = actual_speedup / theoretical_speedup
            if efficiency >= 0.8:
                best_degree = degree
            else:
                break  # if efficiency drops below threshold, stop checking further

        return best_degree


if __name__ == "__main__":
    import torch

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"

    device_name = "NVIDIA A100-SXM4-40GB"

    model_name = "stable-diffusion-3-medium-diffusers"
    model_name = "sd3"
    # model_name = "CogVideoX1.5-5B"
    profiler = ModelProfiler(model_name, device_name)
    profiler.load_cache()
    # profiler.save_cache()
    profiler.compute_all_best_batchsizes()

    print(profiler.best_batchsize)

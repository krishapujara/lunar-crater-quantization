"""
Rigorous Scientific Benchmarking Suite for FP32 vs INT8 Models.
Measures latency distributions (median, P90, P95), throughput (FPS), memory, and size reduction.
"""

import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import psutil

from ..models.inference import CraterDetector
from ..utils.logging import get_logger

logger = get_logger("benchmarking")


@dataclass
class BenchmarkMetrics:
    model_name: str
    model_type: str
    runtime: str
    device: str
    repetition_count: int
    warmup_count: int
    file_size_mb: float
    mean_latency_ms: float
    median_latency_ms: float
    std_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    mean_prep_ms: float
    mean_post_ms: float
    throughput_fps: float
    process_memory_mb: float
    latencies: List[float]


@dataclass
class ComparisonReport:
    fp32_metrics: BenchmarkMetrics
    int8_metrics: BenchmarkMetrics
    size_reduction_mb: float
    size_reduction_pct: float
    latency_reduction_pct: float
    speedup_ratio: float
    system_info: Dict[str, Any]


def get_system_hardware_info() -> Dict[str, Any]:
    """Capture authentic hardware specifications for scientific benchmark context."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count() or 1,
        "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
    }


def benchmark_single_model(
    detector: CraterDetector,
    sample_image: Union[np.ndarray, Path, str],
    warmup_runs: int = 5,
    benchmark_runs: int = 30,
) -> BenchmarkMetrics:
    """
    Execute rigorous latency benchmarking on a single model.
    Includes explicit warm-up runs to ensure steady CPU cache and JIT state.
    """
    logger.info(f"Benchmarking {detector.model_type} ({warmup_runs} warm-ups, {benchmark_runs} repetitions)...")

    # Warm-up phase (measurements discarded)
    for _ in range(warmup_runs):
        detector.detect(sample_image)

    # Benchmark repetitions
    latencies_total: List[float] = []
    latencies_infer: List[float] = []
    latencies_prep: List[float] = []
    latencies_post: List[float] = []

    for _ in range(benchmark_runs):
        res = detector.detect(sample_image)
        latencies_total.append(res.total_time_ms)
        latencies_infer.append(res.inference_time_ms)
        latencies_prep.append(res.preprocessing_time_ms)
        latencies_post.append(res.postprocessing_time_ms)

    lat_arr = np.array(latencies_infer)
    file_size_mb = round(detector.model_path.stat().st_size / (1024 * 1024), 2)
    process_mem_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

    mean_lat = float(np.mean(lat_arr))
    median_lat = float(np.median(lat_arr))
    std_lat = float(np.std(lat_arr))
    p90_lat = float(np.percentile(lat_arr, 90))
    p95_lat = float(np.percentile(lat_arr, 95))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    fps = round(1000.0 / max(1e-4, mean_lat), 2)

    metrics = BenchmarkMetrics(
        model_name=detector.model_path.name,
        model_type=detector.model_type,
        runtime="ONNX Runtime" if "ONNX" in detector.model_type else "PyTorch",
        device=detector.device,
        repetition_count=benchmark_runs,
        warmup_count=warmup_runs,
        file_size_mb=file_size_mb,
        mean_latency_ms=round(mean_lat, 2),
        median_latency_ms=round(median_lat, 2),
        std_latency_ms=round(std_lat, 2),
        p90_latency_ms=round(p90_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        min_latency_ms=round(min_lat, 2),
        max_latency_ms=round(max_lat, 2),
        mean_prep_ms=round(float(np.mean(latencies_prep)), 2),
        mean_post_ms=round(float(np.mean(latencies_post)), 2),
        throughput_fps=fps,
        process_memory_mb=process_mem_mb,
        latencies=[round(float(x), 2) for x in latencies_infer],
    )

    logger.info(
        f"{detector.model_type} Benchmark complete: Median Latency={metrics.median_latency_ms} ms, "
        f"P95={metrics.p95_latency_ms} ms, Throughput={metrics.throughput_fps} FPS, Size={file_size_mb} MB"
    )
    return metrics


def compare_fp32_vs_int8(
    fp32_detector: CraterDetector,
    int8_detector: CraterDetector,
    sample_image: Union[np.ndarray, Path, str],
    warmup_runs: int = 5,
    benchmark_runs: int = 30,
) -> ComparisonReport:
    """Run side-by-side comparative benchmarking between FP32 baseline and INT8 candidate."""
    fp32_m = benchmark_single_model(fp32_detector, sample_image, warmup_runs, benchmark_runs)
    int8_m = benchmark_single_model(int8_detector, sample_image, warmup_runs, benchmark_runs)

    size_diff_mb = round(fp32_m.file_size_mb - int8_m.file_size_mb, 2)
    size_red_pct = round(((fp32_m.file_size_mb - int8_m.file_size_mb) / max(1e-4, fp32_m.file_size_mb)) * 100.0, 2)

    lat_diff = fp32_m.median_latency_ms - int8_m.median_latency_ms
    lat_red_pct = round((lat_diff / max(1e-4, fp32_m.median_latency_ms)) * 100.0, 2)
    speedup = round(fp32_m.median_latency_ms / max(1e-4, int8_m.median_latency_ms), 2)

    report = ComparisonReport(
        fp32_metrics=fp32_m,
        int8_metrics=int8_m,
        size_reduction_mb=size_diff_mb,
        size_reduction_pct=size_red_pct,
        latency_reduction_pct=lat_red_pct,
        speedup_ratio=speedup,
        system_info=get_system_hardware_info(),
    )
    return report

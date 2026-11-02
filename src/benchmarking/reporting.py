"""
Benchmark reporting and data serialization utilities.
Generates CSV and JSON records for analysis and dashboard visualization.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from .benchmark import BenchmarkMetrics, ComparisonReport


def save_benchmark_report(
    report: ComparisonReport,
    output_dir: Path,
    report_name: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Save comparison report as structured JSON and summary CSV.
    Returns: (json_path, csv_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = report_name or f"benchmark_comparison_{timestamp_str}"

    json_path = output_dir / f"{name}.json"
    csv_path = output_dir / f"{name}.csv"

    # JSON export
    data = asdict(report)
    data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # CSV summary export
    rows = [
        {
            "Metric": "Model Format / Runtime",
            "FP32 Baseline": f"{report.fp32_metrics.model_type} ({report.fp32_metrics.runtime})",
            "INT8 Quantized": f"{report.int8_metrics.model_type} ({report.int8_metrics.runtime})",
            "Change / Benefit": "-",
        },
        {
            "Metric": "Serialized Size (MB)",
            "FP32 Baseline": f"{report.fp32_metrics.file_size_mb:.2f} MB",
            "INT8 Quantized": f"{report.int8_metrics.file_size_mb:.2f} MB",
            "Change / Benefit": f"-{report.size_reduction_pct:.1f}% ({report.size_reduction_mb:.2f} MB saved)",
        },
        {
            "Metric": "Median Latency (ms)",
            "FP32 Baseline": f"{report.fp32_metrics.median_latency_ms:.2f} ms",
            "INT8 Quantized": f"{report.int8_metrics.median_latency_ms:.2f} ms",
            "Change / Benefit": f"{report.speedup_ratio:.2f}x speedup ({report.latency_reduction_pct:.1f}% faster)",
        },
        {
            "Metric": "P95 Latency (ms)",
            "FP32 Baseline": f"{report.fp32_metrics.p95_latency_ms:.2f} ms",
            "INT8 Quantized": f"{report.int8_metrics.p95_latency_ms:.2f} ms",
            "Change / Benefit": f"{report.fp32_metrics.p95_latency_ms - report.int8_metrics.p95_latency_ms:.2f} ms diff",
        },
        {
            "Metric": "Throughput (FPS)",
            "FP32 Baseline": f"{report.fp32_metrics.throughput_fps:.1f} FPS",
            "INT8 Quantized": f"{report.int8_metrics.throughput_fps:.1f} FPS",
            "Change / Benefit": f"+{report.int8_metrics.throughput_fps - report.fp32_metrics.throughput_fps:.1f} FPS",
        },
        {
            "Metric": "Device / Platform",
            "FP32 Baseline": f"{report.system_info['platform']}",
            "INT8 Quantized": f"{report.system_info['platform']}",
            "Change / Benefit": f"{report.system_info['cpu_count_logical']} CPUs",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    return json_path, csv_path

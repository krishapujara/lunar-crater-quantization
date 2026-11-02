"""
CLI Script: Benchmark FP32 Baseline vs INT8 Quantized Model on Held-out Test Data.
Usage: python scripts/benchmark_models.py [--repetitions N] [--warmup W]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarking.benchmark import compare_fp32_vs_int8
from src.benchmarking.reporting import save_benchmark_report
from src.config import load_config
from src.data.dataset import LunarCraterDataset
from src.models.inference import CraterDetector
from src.storage.database import DatabaseManager
from src.storage.experiments import ExperimentRegistry
from src.training.evaluate import evaluate_model_on_dataset
from src.utils.logging import get_logger

logger = get_logger("cli_benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs INT8 Crater Detectors")
    parser.add_argument("--repetitions", type=int, default=30, help="Number of benchmark repetitions")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up runs")
    args = parser.parse_args()

    cfg = load_config()

    fp32_onnx = cfg.paths.quantized_dir / "crater_detector_fp32.onnx"
    int8_onnx = cfg.paths.quantized_dir / "crater_detector_int8.onnx"
    pth_model = cfg.paths.checkpoints_dir / "crater_detector_baseline_best.pth"

    # Use FP32 ONNX if available, else PyTorch .pth
    fp32_path = fp32_onnx if fp32_onnx.is_file() else pth_model

    if not fp32_path.is_file():
        logger.error(f"FP32 baseline not found at {fp32_path}. Please train or export model first.")
        sys.exit(1)

    if not int8_onnx.is_file():
        logger.error(f"INT8 model not found at {int8_onnx}. Please run 'python scripts/quantize_model.py' first.")
        sys.exit(1)

    logger.info("Initializing FP32 and INT8 CraterDetectors...")
    fp32_det = CraterDetector(fp32_path)
    int8_det = CraterDetector(int8_onnx)

    # Load test split for evaluation
    split_manifest_file = cfg.paths.splits_dir / "split_manifest.json"
    if not split_manifest_file.is_file():
        logger.error(f"Split manifest not found at {split_manifest_file}.")
        sys.exit(1)

    with open(split_manifest_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    test_records = split_data["splits"]["test"]
    test_dataset = LunarCraterDataset(
        records=test_records,
        base_dir=cfg.paths.data_dir,
        target_size=tuple(cfg.model.input_resolution),
    )

    logger.info(f"Step 1: Evaluating FP32 Detector on {len(test_dataset)} test tiles...")
    fp32_eval = evaluate_model_on_dataset(fp32_det, test_dataset)

    logger.info(f"Step 2: Evaluating INT8 Detector on {len(test_dataset)} test tiles...")
    int8_eval = evaluate_model_on_dataset(int8_det, test_dataset)

    logger.info(f"Step 3: Running Latency & Throughput Benchmark ({args.warmup} warm-ups, {args.repetitions} reps)...")
    sample_img, _ = test_dataset[0]
    sample_arr = (sample_img.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")

    comp_report = compare_fp32_vs_int8(
        fp32_detector=fp32_det,
        int8_detector=int8_det,
        sample_image=sample_arr,
        warmup_runs=args.warmup,
        benchmark_runs=args.repetitions,
    )

    # Save benchmark report files
    json_path, csv_path = save_benchmark_report(comp_report, cfg.paths.benchmarks_dir)

    # Record to Database
    db_mgr = DatabaseManager(cfg.paths.experiments_db, cfg.database_url)
    registry = ExperimentRegistry(db_mgr)

    exp_id = registry.record_experiment(
        model_architecture=cfg.model.architecture,
        model_format="INT8_ONNX vs FP32",
        serialized_size_mb=comp_report.int8_metrics.file_size_mb,
        ap_50=int8_eval.get("ap_50"),
        map_50_95=int8_eval.get("map_50_95"),
        precision=int8_eval.get("precision"),
        recall=int8_eval.get("recall"),
        median_latency_ms=comp_report.int8_metrics.median_latency_ms,
        p95_latency_ms=comp_report.int8_metrics.p95_latency_ms,
        throughput_fps=comp_report.int8_metrics.throughput_fps,
        quantization_method="PTQ Dynamic QInt8",
        hardware_platform=comp_report.system_info.get("platform"),
        raw_metrics={"fp32_eval": fp32_eval, "int8_eval": int8_eval},
        raw_benchmark={
            "size_reduction_pct": comp_report.size_reduction_pct,
            "speedup_ratio": comp_report.speedup_ratio,
            "fp32_latency": comp_report.fp32_metrics.median_latency_ms,
            "int8_latency": comp_report.int8_metrics.median_latency_ms,
        },
    )

    print(f"\n=======================================================")
    print(f"       BENCHMARK AND COMPARISON RESULTS (ID: {exp_id})")
    print(f"=======================================================")
    print(f"Metric                    | FP32 Baseline     | INT8 Quantized")
    print(f"--------------------------+-------------------+-------------------")
    print(f"Serialized Model Size     | {comp_report.fp32_metrics.file_size_mb:6.2f} MB        | {comp_report.int8_metrics.file_size_mb:6.2f} MB (-{comp_report.size_reduction_pct:.1f}%)")
    print(f"Median Inference Latency  | {comp_report.fp32_metrics.median_latency_ms:6.2f} ms        | {comp_report.int8_metrics.median_latency_ms:6.2f} ms ({comp_report.speedup_ratio:.2f}x speedup)")
    print(f"P95 Inference Latency     | {comp_report.fp32_metrics.p95_latency_ms:6.2f} ms        | {comp_report.int8_metrics.p95_latency_ms:6.2f} ms")
    print(f"Throughput (FPS)          | {comp_report.fp32_metrics.throughput_fps:6.1f} FPS       | {comp_report.int8_metrics.throughput_fps:6.1f} FPS")
    print(f"mAP @ IoU 0.50            | {fp32_eval['ap_50']:6.4f}           | {int8_eval['ap_50']:6.4f} (diff: {int8_eval['ap_50'] - fp32_eval['ap_50']:+.4f})")
    print(f"mAP @ IoU 0.50:0.95       | {fp32_eval['map_50_95']:6.4f}           | {int8_eval['map_50_95']:6.4f} (diff: {int8_eval['map_50_95'] - fp32_eval['map_50_95']:+.4f})")
    print(f"Precision                 | {fp32_eval['precision']:6.4f}           | {int8_eval['precision']:6.4f}")
    print(f"Recall                    | {fp32_eval['recall']:6.4f}           | {int8_eval['recall']:6.4f}")
    print(f"=======================================================")
    print(f"Report saved to:\n  JSON: {json_path}\n  CSV:  {csv_path}")


if __name__ == "__main__":
    main()

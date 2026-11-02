"""
Project Configuration Loader and Settings Dataclasses.
Loads YAML configurations, handles environment overrides, and resolves paths.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ProjectSettings:
    name: str = "lunar-crater-quantization"
    title: str = "Micro-Quantized CNN for Onboard Crater Detection â€” Chandrayaan-Inspired Lunar Research"
    version: str = "1.0.0"
    random_seed: int = 42


@dataclass
class PathSettings:
    root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    splits_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "splits")
    samples_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "samples")
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    checkpoints_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "checkpoints")
    quantized_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "quantized")
    benchmarks_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "benchmarks")
    experiments_db: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "experiments.db")

    def ensure_directories(self) -> None:
        """Create all required directories if they don't already exist."""
        for p in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.splits_dir,
            self.samples_dir,
            self.artifacts_dir,
            self.checkpoints_dir,
            self.quantized_dir,
            self.benchmarks_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class DataSettings:
    dataset_name: str = "LRO_NAC_ChangE4"
    source_repository: str = "https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data"
    instrument: str = "Lunar Reconnaissance Orbiter (LRO) Narrow Angle Camera (NAC)"
    target_region: str = "Von KÃ¡rmÃ¡n crater / Chang'e-4 landing zone (45Â°-46Â° S, 176.4Â°-178.8Â° E)"
    image_size: int = 320
    min_crater_diameter_px: float = 6.0
    split_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15})


@dataclass
class ModelSettings:
    architecture: str = "ssdlite320_mobilenet_v3_large"
    backbone: str = "MobileNetV3-Large"
    num_classes: int = 2
    pretrained_backbone: bool = True
    input_channels: int = 3
    input_resolution: List[int] = field(default_factory=lambda: [320, 320])


@dataclass
class TrainingSettings:
    batch_size: int = 4
    epochs: int = 8
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    momentum: float = 0.9
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    early_stopping_patience: int = 3
    device: str = "cpu"
    seed: int = 42


@dataclass
class QuantizationSettings:
    method: str = "onnx_dynamic_int8"
    runtime: str = "onnxruntime"
    opset_version: int = 14
    weight_type: str = "QInt8"
    target_hardware: str = "Radiation-Tolerant Spaceborne Embedded CPU / DSP (e.g., LEON4-FT / ARM Cortex)"


@dataclass
class EvaluationSettings:
    iou_threshold: float = 0.50
    confidence_threshold: float = 0.35
    metric_iou_range: List[float] = field(default_factory=lambda: [0.50, 0.95, 0.05])


@dataclass
class BenchmarkSettings:
    warmup_runs: int = 5
    benchmark_runs: int = 30
    batch_size: int = 1


@dataclass
class AppConfig:
    project: ProjectSettings = field(default_factory=ProjectSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    quantization: QuantizationSettings = field(default_factory=QuantizationSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    benchmarking: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    database_url: Optional[str] = None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML file and apply environment variable overrides."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "default_config.yaml"

    cfg = AppConfig()

    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Project
        if "project" in raw:
            p = raw["project"]
            cfg.project = ProjectSettings(
                name=p.get("name", cfg.project.name),
                title=p.get("title", cfg.project.title),
                version=p.get("version", cfg.project.version),
                random_seed=p.get("random_seed", cfg.project.random_seed),
            )

        # Paths
        if "paths" in raw:
            pts = raw["paths"]
            cfg.paths = PathSettings(
                data_dir=PROJECT_ROOT / pts.get("data_dir", "data"),
                raw_data_dir=PROJECT_ROOT / pts.get("raw_data_dir", "data/raw"),
                processed_data_dir=PROJECT_ROOT / pts.get("processed_data_dir", "data/processed"),
                splits_dir=PROJECT_ROOT / pts.get("splits_dir", "data/splits"),
                samples_dir=PROJECT_ROOT / pts.get("samples_dir", "data/samples"),
                artifacts_dir=PROJECT_ROOT / pts.get("artifacts_dir", "artifacts"),
                checkpoints_dir=PROJECT_ROOT / pts.get("checkpoints_dir", "artifacts/checkpoints"),
                quantized_dir=PROJECT_ROOT / pts.get("quantized_dir", "artifacts/quantized"),
                benchmarks_dir=PROJECT_ROOT / pts.get("benchmarks_dir", "artifacts/benchmarks"),
                experiments_db=PROJECT_ROOT / pts.get("experiments_db", "artifacts/experiments.db"),
            )

        # Data
        if "data" in raw:
            d = raw["data"]
            cfg.data = DataSettings(
                dataset_name=d.get("dataset_name", cfg.data.dataset_name),
                source_repository=d.get("source_repository", cfg.data.source_repository),
                instrument=d.get("instrument", cfg.data.instrument),
                target_region=d.get("target_region", cfg.data.target_region),
                image_size=d.get("image_size", cfg.data.image_size),
                min_crater_diameter_px=d.get("min_crater_diameter_px", cfg.data.min_crater_diameter_px),
                split_ratios=d.get("split_ratios", cfg.data.split_ratios),
            )

        # Model
        if "model" in raw:
            m = raw["model"]
            cfg.model = ModelSettings(
                architecture=m.get("architecture", cfg.model.architecture),
                backbone=m.get("backbone", cfg.model.backbone),
                num_classes=m.get("num_classes", cfg.model.num_classes),
                pretrained_backbone=m.get("pretrained_backbone", cfg.model.pretrained_backbone),
                input_channels=m.get("input_channels", cfg.model.input_channels),
                input_resolution=m.get("input_resolution", cfg.model.input_resolution),
            )

        # Training
        if "training" in raw:
            t = raw["training"]
            cfg.training = TrainingSettings(
                batch_size=t.get("batch_size", cfg.training.batch_size),
                epochs=t.get("epochs", cfg.training.epochs),
                learning_rate=t.get("learning_rate", cfg.training.learning_rate),
                weight_decay=t.get("weight_decay", cfg.training.weight_decay),
                momentum=t.get("momentum", cfg.training.momentum),
                optimizer=t.get("optimizer", cfg.training.optimizer),
                lr_scheduler=t.get("lr_scheduler", cfg.training.lr_scheduler),
                early_stopping_patience=t.get("early_stopping_patience", cfg.training.early_stopping_patience),
                device=t.get("device", cfg.training.device),
                seed=t.get("seed", cfg.training.seed),
            )

        # Quantization
        if "quantization" in raw:
            q = raw["quantization"]
            cfg.quantization = QuantizationSettings(
                method=q.get("method", cfg.quantization.method),
                runtime=q.get("runtime", cfg.quantization.runtime),
                opset_version=q.get("opset_version", cfg.quantization.opset_version),
                weight_type=q.get("weight_type", cfg.quantization.weight_type),
                target_hardware=q.get("target_hardware", cfg.quantization.target_hardware),
            )

        # Evaluation
        if "evaluation" in raw:
            ev = raw["evaluation"]
            cfg.evaluation = EvaluationSettings(
                iou_threshold=ev.get("iou_threshold", cfg.evaluation.iou_threshold),
                confidence_threshold=ev.get("confidence_threshold", cfg.evaluation.confidence_threshold),
                metric_iou_range=ev.get("metric_iou_range", cfg.evaluation.metric_iou_range),
            )

        # Benchmarking
        if "benchmarking" in raw:
            b = raw["benchmarking"]
            cfg.benchmarking = BenchmarkSettings(
                warmup_runs=b.get("warmup_runs", cfg.benchmarking.warmup_runs),
                benchmark_runs=b.get("benchmark_runs", cfg.benchmarking.benchmark_runs),
                batch_size=b.get("batch_size", cfg.benchmarking.batch_size),
            )

    # Environment overrides
    db_env = os.environ.get("DATABASE_URL")
    if db_env and db_env.strip():
        cfg.database_url = db_env.strip()

    if "DEFAULT_DEVICE" in os.environ:
        cfg.training.device = os.environ["DEFAULT_DEVICE"]
    if "DEFAULT_BATCH_SIZE" in os.environ:
        try:
            cfg.training.batch_size = int(os.environ["DEFAULT_BATCH_SIZE"])
        except ValueError:
            pass
    if "DEFAULT_EPOCHS" in os.environ:
        try:
            cfg.training.epochs = int(os.environ["DEFAULT_EPOCHS"])
        except ValueError:
            pass

    cfg.paths.ensure_directories()
    return cfg


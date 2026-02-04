#!/usr/bin/env python3
"""
Training script for fine-tuning predator detection models.
Supports YOLOv8 fine-tuning with custom datasets.
"""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class PredatorModelTrainer:
    """
    Fine-tunes YOLO models for predator detection.
    """

    # Predator classes for the garden sentinel system
    PREDATOR_CLASSES = [
        "fox",
        "badger",
        "cat",
        "dog",
        "hawk",
        "eagle",
        "owl",
        "crow",
        "magpie",
        "rat",
        "weasel",
        "stoat",
        "mink",
    ]

    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "models",
        base_model: str = "yolov8n.pt",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.base_model = base_model

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        val_split: float = 0.2,
    ) -> Path:
        """
        Prepare dataset in YOLO format.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format labels
            val_split: Fraction of data for validation

        Returns:
            Path to the dataset YAML file
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        # Create dataset directory structure
        dataset_dir = self.data_dir / "dataset"
        train_images = dataset_dir / "train" / "images"
        train_labels = dataset_dir / "train" / "labels"
        val_images = dataset_dir / "val" / "images"
        val_labels = dataset_dir / "val" / "labels"

        for d in [train_images, train_labels, val_images, val_labels]:
            d.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        logger.info(f"Found {len(image_files)} images")

        # Split into train/val
        import random
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - val_split))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        # Copy files
        for files, img_dir, lbl_dir in [
            (train_files, train_images, train_labels),
            (val_files, val_images, val_labels),
        ]:
            for img_file in files:
                # Copy image
                shutil.copy(img_file, img_dir / img_file.name)

                # Copy label if exists
                label_file = labels_path / img_file.with_suffix(".txt").name
                if label_file.exists():
                    shutil.copy(label_file, lbl_dir / label_file.name)

        logger.info(f"Dataset prepared: {len(train_files)} train, {len(val_files)} val")

        # Create dataset YAML
        dataset_yaml = dataset_dir / "dataset.yaml"
        dataset_config = {
            "path": str(dataset_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "names": {i: name for i, name in enumerate(self.PREDATOR_CLASSES)},
        }

        with open(dataset_yaml, "w") as f:
            yaml.dump(dataset_config, f)

        return dataset_yaml

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        device: str = "0",  # GPU device, or "cpu"
        patience: int = 20,
        workers: int = 8,
        resume: bool = False,
        freeze_layers: int = 0,
    ) -> Path:
        """
        Train/fine-tune the model.

        Args:
            dataset_yaml: Path to dataset configuration
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            device: Device to train on ("0" for GPU 0, "cpu" for CPU)
            patience: Early stopping patience
            workers: Number of data loading workers
            resume: Resume from last checkpoint
            freeze_layers: Number of backbone layers to freeze

        Returns:
            Path to the trained model
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

        # Load base model
        model = YOLO(self.base_model)
        logger.info(f"Loaded base model: {self.base_model}")

        # Training run name
        run_name = f"predator_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Train
        logger.info(f"Starting training for {epochs} epochs")
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            patience=patience,
            workers=workers,
            project=str(self.output_dir),
            name=run_name,
            resume=resume,
            freeze=freeze_layers,
            # Augmentation settings for outdoor scenes
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
        )

        # Get best model path
        best_model = self.output_dir / run_name / "weights" / "best.pt"
        logger.info(f"Training complete. Best model: {best_model}")

        return best_model

    def export_model(
        self,
        model_path: str,
        format: str = "onnx",
        img_size: int = 640,
        half: bool = False,
    ) -> Path:
        """
        Export model to different formats.

        Supported formats:
        - onnx: ONNX format
        - tflite: TensorFlow Lite (for Coral TPU)
        - engine: TensorRT (for NVIDIA)
        - edgetpu: Edge TPU (for Coral)

        Args:
            model_path: Path to the trained model
            format: Export format
            img_size: Input image size
            half: Use FP16 quantization

        Returns:
            Path to exported model
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed")
            raise

        model = YOLO(model_path)

        logger.info(f"Exporting model to {format} format")
        exported = model.export(
            format=format,
            imgsz=img_size,
            half=half,
        )

        logger.info(f"Model exported: {exported}")
        return Path(exported)

    def export_for_coral(self, model_path: str, img_size: int = 640) -> Path:
        """
        Export model for Google Coral Edge TPU.

        This is a two-step process:
        1. Export to TFLite
        2. Compile for Edge TPU (requires edgetpu_compiler)
        """
        # First export to TFLite
        tflite_path = self.export_model(model_path, format="tflite", img_size=img_size)

        # Try to compile for Edge TPU
        try:
            import subprocess
            result = subprocess.run(
                ["edgetpu_compiler", str(tflite_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                edgetpu_model = tflite_path.with_name(
                    tflite_path.stem + "_edgetpu.tflite"
                )
                logger.info(f"Edge TPU model: {edgetpu_model}")
                return edgetpu_model
            else:
                logger.warning(f"Edge TPU compilation failed: {result.stderr}")
        except FileNotFoundError:
            logger.warning("edgetpu_compiler not found. Install Edge TPU runtime.")

        return tflite_path

    def validate(self, model_path: str, dataset_yaml: str) -> dict:
        """
        Validate model performance.

        Returns:
            Dictionary with validation metrics
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed")
            raise

        model = YOLO(model_path)

        logger.info("Running validation")
        results = model.val(data=dataset_yaml)

        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }

        logger.info(f"Validation results: {metrics}")
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Train predator detection model")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Prepare dataset command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prepare_parser.add_argument("--images", required=True, help="Images directory")
    prepare_parser.add_argument("--labels", required=True, help="Labels directory")
    prepare_parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--dataset", required=True, help="Dataset YAML path")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--img-size", type=int, default=640, help="Image size")
    train_parser.add_argument("--device", default="0", help="Device (0 for GPU, cpu for CPU)")
    train_parser.add_argument("--base-model", default="yolov8n.pt", help="Base model")
    train_parser.add_argument("--resume", action="store_true", help="Resume training")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--model", required=True, help="Model path")
    export_parser.add_argument("--format", default="onnx", help="Export format")
    export_parser.add_argument("--coral", action="store_true", help="Export for Coral TPU")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate model")
    val_parser.add_argument("--model", required=True, help="Model path")
    val_parser.add_argument("--dataset", required=True, help="Dataset YAML path")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    trainer = PredatorModelTrainer(
        base_model=getattr(args, "base_model", "yolov8n.pt")
    )

    if args.command == "prepare":
        trainer.prepare_dataset(args.images, args.labels, args.val_split)

    elif args.command == "train":
        trainer.train(
            dataset_yaml=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            resume=args.resume,
        )

    elif args.command == "export":
        if args.coral:
            trainer.export_for_coral(args.model)
        else:
            trainer.export_model(args.model, format=args.format)

    elif args.command == "validate":
        trainer.validate(args.model, args.dataset)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

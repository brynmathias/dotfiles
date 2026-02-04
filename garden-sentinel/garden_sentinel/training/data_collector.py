#!/usr/bin/env python3
"""
Data collection and labeling tools for Garden Sentinel.
Helps collect and annotate training data from the system.
"""

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects and organizes training data from the Garden Sentinel system.
    """

    PREDATOR_CLASSES = [
        "fox", "badger", "cat", "dog", "hawk", "eagle", "owl",
        "crow", "magpie", "rat", "weasel", "stoat", "mink",
    ]

    def __init__(self, output_dir: str = "data/training"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.metadata_file = self.output_dir / "metadata.json"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load or create metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"samples": [], "class_counts": {}}

    def _save_metadata(self):
        """Save metadata file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def import_from_events(self, events_dir: str, min_confidence: float = 0.5):
        """
        Import training data from saved events.

        Args:
            events_dir: Directory containing event folders
            min_confidence: Minimum detection confidence to include
        """
        events_path = Path(events_dir)
        imported = 0

        for event_dir in events_path.iterdir():
            if not event_dir.is_dir():
                continue

            event_file = event_dir / "event.json"
            frame_file = event_dir / "frame.jpg"

            if not event_file.exists() or not frame_file.exists():
                continue

            with open(event_file) as f:
                event_data = json.load(f)

            # Load frame
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue

            height, width = frame.shape[:2]

            # Process detections
            detections = event_data.get("detections", [])
            valid_detections = [
                d for d in detections
                if d.get("confidence", 0) >= min_confidence
            ]

            if not valid_detections:
                continue

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_name = f"{timestamp}.jpg"
            label_name = f"{timestamp}.txt"

            # Save image
            cv2.imwrite(str(self.images_dir / image_name), frame)

            # Create YOLO format labels
            labels = []
            for det in valid_detections:
                class_name = det.get("class_name", "").lower()

                # Map to class ID
                if class_name in self.PREDATOR_CLASSES:
                    class_id = self.PREDATOR_CLASSES.index(class_name)
                else:
                    continue  # Skip unknown classes

                bbox = det.get("bbox", {})
                x_center = bbox.get("x", 0) + bbox.get("width", 0) / 2
                y_center = bbox.get("y", 0) + bbox.get("height", 0) / 2
                w = bbox.get("width", 0)
                h = bbox.get("height", 0)

                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                # Update class counts
                if class_name not in self.metadata["class_counts"]:
                    self.metadata["class_counts"][class_name] = 0
                self.metadata["class_counts"][class_name] += 1

            # Save labels
            with open(self.labels_dir / label_name, "w") as f:
                f.write("\n".join(labels))

            # Update metadata
            self.metadata["samples"].append({
                "image": image_name,
                "label": label_name,
                "source_event": event_dir.name,
                "num_detections": len(valid_detections),
                "classes": [d.get("class_name") for d in valid_detections],
            })

            imported += 1

        self._save_metadata()
        logger.info(f"Imported {imported} samples from events")

    def import_external_dataset(
        self,
        images_dir: str,
        labels_dir: Optional[str] = None,
        class_mapping: Optional[dict] = None,
    ):
        """
        Import external dataset (e.g., from Roboflow, COCO subset).

        Args:
            images_dir: Directory with images
            labels_dir: Directory with YOLO format labels (optional)
            class_mapping: Map external class IDs to our classes
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir) if labels_dir else None

        imported = 0

        for img_file in images_path.glob("*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Copy image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            new_name = f"ext_{timestamp}{img_file.suffix}"
            shutil.copy(img_file, self.images_dir / new_name)

            # Copy/convert label if exists
            if labels_path:
                label_file = labels_path / img_file.with_suffix(".txt").name
                if label_file.exists():
                    if class_mapping:
                        # Remap classes
                        with open(label_file) as f:
                            lines = f.readlines()

                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                old_class_id = int(parts[0])
                                if old_class_id in class_mapping:
                                    new_class = class_mapping[old_class_id]
                                    if new_class in self.PREDATOR_CLASSES:
                                        new_class_id = self.PREDATOR_CLASSES.index(new_class)
                                        parts[0] = str(new_class_id)
                                        new_lines.append(" ".join(parts))

                        with open(self.labels_dir / f"ext_{timestamp}.txt", "w") as f:
                            f.write("\n".join(new_lines))
                    else:
                        shutil.copy(label_file, self.labels_dir / f"ext_{timestamp}.txt")

            imported += 1

        self._save_metadata()
        logger.info(f"Imported {imported} external samples")

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.metadata["samples"]),
            "class_counts": self.metadata["class_counts"],
            "classes": self.PREDATOR_CLASSES,
        }

        # Count labeled vs unlabeled
        labeled = sum(
            1 for s in self.metadata["samples"]
            if (self.labels_dir / s["label"]).exists()
        )
        stats["labeled_samples"] = labeled
        stats["unlabeled_samples"] = stats["total_samples"] - labeled

        return stats

    def create_class_yaml(self) -> Path:
        """Create a YAML file with class definitions."""
        yaml_path = self.output_dir / "classes.yaml"

        content = "names:\n"
        for i, name in enumerate(self.PREDATOR_CLASSES):
            content += f"  {i}: {name}\n"

        with open(yaml_path, "w") as f:
            f.write(content)

        return yaml_path


class LabelingTool:
    """
    Simple labeling tool for annotating images.
    """

    def __init__(self, data_dir: str = "data/training"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"

        self.classes = DataCollector.PREDATOR_CLASSES
        self.current_class = 0
        self.current_image_idx = 0
        self.image_files = []
        self.current_boxes = []
        self.drawing = False
        self.start_point = None

    def load_images(self):
        """Load list of images to label."""
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png"))
        )
        logger.info(f"Found {len(self.image_files)} images")

    def load_existing_labels(self, image_path: Path) -> list:
        """Load existing labels for an image."""
        label_path = self.labels_dir / image_path.with_suffix(".txt").name
        boxes = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:5])
                        boxes.append({
                            "class_id": class_id,
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": w,
                            "height": h,
                        })

        return boxes

    def save_labels(self, image_path: Path, boxes: list):
        """Save labels for an image."""
        label_path = self.labels_dir / image_path.with_suffix(".txt").name

        with open(label_path, "w") as f:
            for box in boxes:
                f.write(
                    f"{box['class_id']} {box['x_center']:.6f} "
                    f"{box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}\n"
                )

    def run_interactive(self):
        """Run interactive labeling tool using OpenCV."""
        self.load_images()

        if not self.image_files:
            logger.error("No images found to label")
            return

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                end_point = (x, y)

                # Calculate normalized coordinates
                img_h, img_w = param["frame"].shape[:2]
                x1 = min(self.start_point[0], end_point[0]) / img_w
                y1 = min(self.start_point[1], end_point[1]) / img_h
                x2 = max(self.start_point[0], end_point[0]) / img_w
                y2 = max(self.start_point[1], end_point[1]) / img_h

                self.current_boxes.append({
                    "class_id": self.current_class,
                    "x_center": (x1 + x2) / 2,
                    "y_center": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                })

        cv2.namedWindow("Labeling Tool")

        while True:
            image_path = self.image_files[self.current_image_idx]
            frame = cv2.imread(str(image_path))

            if frame is None:
                self.current_image_idx += 1
                continue

            self.current_boxes = self.load_existing_labels(image_path)

            cv2.setMouseCallback("Labeling Tool", mouse_callback, {"frame": frame})

            while True:
                display = frame.copy()
                h, w = display.shape[:2]

                # Draw existing boxes
                for box in self.current_boxes:
                    x1 = int((box["x_center"] - box["width"] / 2) * w)
                    y1 = int((box["y_center"] - box["height"] / 2) * h)
                    x2 = int((box["x_center"] + box["width"] / 2) * w)
                    y2 = int((box["y_center"] + box["height"] / 2) * h)

                    color = (0, 255, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display,
                        self.classes[box["class_id"]],
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                # Draw info
                info = [
                    f"Image: {self.current_image_idx + 1}/{len(self.image_files)}",
                    f"Class: {self.classes[self.current_class]} ({self.current_class})",
                    f"Boxes: {len(self.current_boxes)}",
                    "",
                    "Controls:",
                    "  q/w - Change class",
                    "  a/d - Prev/Next image",
                    "  s - Save labels",
                    "  z - Undo last box",
                    "  ESC - Exit",
                ]

                for i, line in enumerate(info):
                    cv2.putText(display, line, (10, 20 + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Labeling Tool", display)

                key = cv2.waitKey(30) & 0xFF

                if key == 27:  # ESC
                    cv2.destroyAllWindows()
                    return
                elif key == ord("q"):
                    self.current_class = max(0, self.current_class - 1)
                elif key == ord("w"):
                    self.current_class = min(len(self.classes) - 1, self.current_class + 1)
                elif key == ord("a"):
                    self.save_labels(image_path, self.current_boxes)
                    self.current_image_idx = max(0, self.current_image_idx - 1)
                    break
                elif key == ord("d"):
                    self.save_labels(image_path, self.current_boxes)
                    self.current_image_idx = min(
                        len(self.image_files) - 1, self.current_image_idx + 1
                    )
                    break
                elif key == ord("s"):
                    self.save_labels(image_path, self.current_boxes)
                    logger.info(f"Saved labels for {image_path.name}")
                elif key == ord("z") and self.current_boxes:
                    self.current_boxes.pop()


def main():
    parser = argparse.ArgumentParser(description="Data collection and labeling tools")
    subparsers = parser.add_subparsers(dest="command")

    # Import from events
    import_parser = subparsers.add_parser("import-events", help="Import from system events")
    import_parser.add_argument("--events-dir", required=True, help="Events directory")
    import_parser.add_argument("--min-confidence", type=float, default=0.5)

    # Import external dataset
    ext_parser = subparsers.add_parser("import-external", help="Import external dataset")
    ext_parser.add_argument("--images", required=True, help="Images directory")
    ext_parser.add_argument("--labels", help="Labels directory")

    # Statistics
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")

    # Labeling tool
    label_parser = subparsers.add_parser("label", help="Run labeling tool")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    collector = DataCollector()

    if args.command == "import-events":
        collector.import_from_events(args.events_dir, args.min_confidence)
    elif args.command == "import-external":
        collector.import_external_dataset(args.images, args.labels)
    elif args.command == "stats":
        stats = collector.get_statistics()
        print(json.dumps(stats, indent=2))
    elif args.command == "label":
        tool = LabelingTool()
        tool.run_interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

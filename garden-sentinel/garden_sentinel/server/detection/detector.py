"""
Predator detection module using YOLO models.
Supports YOLOv8, custom trained models, and various backends.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from garden_sentinel.shared import BoundingBox, Detection, PredatorType, ThreatLevel, PREDATOR_THREAT_LEVELS


@dataclass
class DetectionConfig:
    model_type: str = "yolov8"
    model_path: str = "models/predator_detector.pt"
    use_pretrained: bool = True
    pretrained_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cuda"
    predator_classes: list[str] = field(default_factory=lambda: ["bird", "cat", "dog"])


# Mapping from COCO classes to predator types
COCO_TO_PREDATOR = {
    "bird": PredatorType.HAWK,  # Generic bird, could be refined
    "cat": PredatorType.CAT,
    "dog": PredatorType.DOG,
    "bear": PredatorType.UNKNOWN_PREDATOR,
}

# Custom class mapping (for fine-tuned models)
CUSTOM_CLASS_MAPPING = {
    "fox": PredatorType.FOX,
    "badger": PredatorType.BADGER,
    "cat": PredatorType.CAT,
    "dog": PredatorType.DOG,
    "hawk": PredatorType.HAWK,
    "eagle": PredatorType.EAGLE,
    "owl": PredatorType.OWL,
    "crow": PredatorType.CROW,
    "magpie": PredatorType.MAGPIE,
    "rat": PredatorType.RAT,
    "weasel": PredatorType.WEASEL,
    "stoat": PredatorType.STOAT,
    "mink": PredatorType.MINK,
}


class PredatorDetector:
    """
    Predator detection using YOLO models.
    Supports both pre-trained COCO models and custom fine-tuned models.
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._model = None
        self._class_names = []
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the detection model."""
        try:
            if self.config.model_type == "yolov8":
                return self._init_yolov8()
            elif self.config.model_type == "yolov5":
                return self._init_yolov5()
            else:
                logger.error(f"Unknown model type: {self.config.model_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            return False

    def _init_yolov8(self) -> bool:
        """Initialize YOLOv8 model."""
        try:
            from ultralytics import YOLO

            # Choose model path
            if self.config.use_pretrained:
                model_path = self.config.pretrained_model
                logger.info(f"Using pre-trained YOLOv8 model: {model_path}")
            else:
                model_path = self.config.model_path
                if not Path(model_path).exists():
                    logger.warning(f"Custom model not found: {model_path}, falling back to pretrained")
                    model_path = self.config.pretrained_model

            self._model = YOLO(model_path)

            # Set device
            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._model.to("cuda")
                    logger.info("Using CUDA for inference")
                else:
                    logger.warning("CUDA not available, using CPU")
            elif self.config.device == "mps":
                import torch
                if torch.backends.mps.is_available():
                    self._model.to("mps")
                    logger.info("Using MPS (Apple Silicon) for inference")

            # Get class names
            self._class_names = self._model.names
            logger.info(f"Model loaded with {len(self._class_names)} classes")

            self._initialized = True
            return True

        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            return False

    def _init_yolov5(self) -> bool:
        """Initialize YOLOv5 model."""
        try:
            import torch

            if self.config.use_pretrained:
                self._model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            else:
                self._model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.config.model_path)

            # Set device
            if self.config.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            elif self.config.device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")

            self._class_names = self._model.names
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to load YOLOv5: {e}")
            return False

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """
        Run detection on a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            tuple: (list of Detection objects, inference time in ms)
        """
        if not self._initialized:
            return [], 0

        start_time = time.time()

        try:
            # Run inference
            results = self._model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )

            inference_time = (time.time() - start_time) * 1000

            # Parse results
            detections = self._parse_results(results, frame.shape[:2])

            return detections, inference_time

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return [], 0

    def _parse_results(self, results, frame_shape: tuple) -> list[Detection]:
        """Parse YOLO results into Detection objects."""
        detections = []
        height, width = frame_shape

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # Get class name
                class_name = self._class_names.get(class_id, f"class_{class_id}")

                # Filter for predator classes
                if (self.config.predator_classes and
                    class_name.lower() not in [c.lower() for c in self.config.predator_classes]):
                    continue

                # Normalize coordinates
                bbox = BoundingBox(
                    x=x1 / width,
                    y=y1 / height,
                    width=(x2 - x1) / width,
                    height=(y2 - y1) / height,
                )

                # Map to predator type
                predator_type = self._map_to_predator_type(class_name)
                threat_level = PREDATOR_THREAT_LEVELS.get(predator_type, ThreatLevel.MEDIUM) if predator_type else None

                detection = Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    predator_type=predator_type,
                    threat_level=threat_level,
                )

                detections.append(detection)

        return detections

    def _map_to_predator_type(self, class_name: str) -> Optional[PredatorType]:
        """Map a class name to a PredatorType."""
        class_lower = class_name.lower()

        # Check custom mapping first
        if class_lower in CUSTOM_CLASS_MAPPING:
            return CUSTOM_CLASS_MAPPING[class_lower]

        # Check COCO mapping
        if class_lower in COCO_TO_PREDATOR:
            return COCO_TO_PREDATOR[class_lower]

        # Try direct enum lookup
        try:
            return PredatorType(class_lower)
        except ValueError:
            return None

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Draw detection boxes on a frame.

        Returns:
            Frame with detections drawn
        """
        result = frame.copy()
        height, width = frame.shape[:2]

        # Color map for threat levels
        colors = {
            ThreatLevel.LOW: (0, 255, 0),       # Green
            ThreatLevel.MEDIUM: (0, 255, 255),   # Yellow
            ThreatLevel.HIGH: (0, 165, 255),     # Orange
            ThreatLevel.CRITICAL: (0, 0, 255),   # Red
        }

        for det in detections:
            # Convert normalized coordinates to pixels
            x1 = int(det.bbox.x * width)
            y1 = int(det.bbox.y * height)
            x2 = int((det.bbox.x + det.bbox.width) * width)
            y2 = int((det.bbox.y + det.bbox.height) * height)

            # Get color based on threat level
            color = colors.get(det.threat_level, (255, 255, 255))

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                # Prepare label
                label = f"{det.class_name}: {det.confidence:.2f}"
                if det.threat_level:
                    label += f" [{det.threat_level.value}]"

                # Draw label background
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    result,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1,
                )

                # Draw label text
                cv2.putText(
                    result,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

        return result

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names.values()) if isinstance(self._class_names, dict) else self._class_names

"""
Edge inference module supporting Google Coral TPU and Hailo AI accelerators.
Performs on-device predator detection to reduce latency and server load.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from garden_sentinel.shared import BoundingBox, Detection


@dataclass
class InferenceConfig:
    enabled: bool = False
    accelerator: str = "coral"  # "coral", "hailo", "cpu"
    model_path: str = "/opt/garden-sentinel/models/predator_detector.tflite"
    confidence_threshold: float = 0.5
    inference_interval_ms: int = 200
    report_detections: bool = True


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        pass

    @abstractmethod
    def run_inference(self, frame: np.ndarray) -> list[Detection]:
        pass

    @abstractmethod
    def cleanup(self):
        pass


class CoralBackend(InferenceBackend):
    """Google Coral TPU inference backend."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._labels = []

    def load_model(self, model_path: str) -> bool:
        try:
            from pycoral.adapters import common, detect
            from pycoral.utils.edgetpu import make_interpreter

            self._interpreter = make_interpreter(model_path)
            self._interpreter.allocate_tensors()

            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            # Load labels if available
            labels_path = Path(model_path).with_suffix(".txt")
            if labels_path.exists():
                self._labels = labels_path.read_text().strip().split("\n")

            logger.info(f"Coral model loaded: {model_path}")
            return True

        except ImportError:
            logger.error("pycoral not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load Coral model: {e}")
            return False

    def run_inference(self, frame: np.ndarray) -> list[Detection]:
        if self._interpreter is None:
            return []

        try:
            from pycoral.adapters import common, detect

            # Resize frame to model input size
            input_shape = self._input_details[0]["shape"]
            height, width = input_shape[1], input_shape[2]

            resized = cv2.resize(frame, (width, height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            common.set_input(self._interpreter, rgb)
            self._interpreter.invoke()

            # Get detections
            objs = detect.get_objects(
                self._interpreter, self.confidence_threshold
            )

            frame_h, frame_w = frame.shape[:2]
            detections = []

            for obj in objs:
                # Normalize bounding box
                bbox = BoundingBox(
                    x=obj.bbox.xmin / width,
                    y=obj.bbox.ymin / height,
                    width=(obj.bbox.xmax - obj.bbox.xmin) / width,
                    height=(obj.bbox.ymax - obj.bbox.ymin) / height,
                )

                class_name = self._labels[obj.id] if obj.id < len(self._labels) else f"class_{obj.id}"

                detections.append(Detection(
                    class_name=class_name,
                    confidence=obj.score,
                    bbox=bbox,
                ))

            return detections

        except Exception as e:
            logger.error(f"Coral inference error: {e}")
            return []

    def cleanup(self):
        self._interpreter = None


class HailoBackend(InferenceBackend):
    """Hailo AI accelerator inference backend."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._hef = None
        self._vdevice = None
        self._network_group = None
        self._labels = []
        self._input_shape = None

    def load_model(self, model_path: str) -> bool:
        try:
            from hailo_platform import VDevice, HEF, ConfigureParams

            # Load HEF model
            self._hef = HEF(model_path)

            # Create virtual device
            self._vdevice = VDevice()

            # Configure network
            configure_params = ConfigureParams.create_from_hef(
                self._hef, interface=self._vdevice.get_default_streams_interface()
            )
            self._network_group = self._vdevice.configure(self._hef, configure_params)[0]

            # Get input shape
            input_vstreams_info = self._network_group.get_input_vstream_infos()
            if input_vstreams_info:
                self._input_shape = input_vstreams_info[0].shape

            # Load labels if available
            labels_path = Path(model_path).with_suffix(".txt")
            if labels_path.exists():
                self._labels = labels_path.read_text().strip().split("\n")

            logger.info(f"Hailo model loaded: {model_path}")
            return True

        except ImportError:
            logger.error("hailo_platform not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load Hailo model: {e}")
            return False

    def run_inference(self, frame: np.ndarray) -> list[Detection]:
        if self._network_group is None:
            return []

        try:
            from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

            # Preprocess
            height, width = self._input_shape[1], self._input_shape[2]
            resized = cv2.resize(frame, (width, height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)

            # Run inference
            input_params = InputVStreamParams.make_from_network_group(
                self._network_group, quantized=True
            )
            output_params = OutputVStreamParams.make_from_network_group(
                self._network_group, quantized=False
            )

            with InferVStreams(self._network_group, input_params, output_params) as infer_pipeline:
                input_dict = {self._network_group.get_input_vstream_infos()[0].name: input_data}
                output = infer_pipeline.infer(input_dict)

            # Parse output (format depends on model)
            detections = self._parse_detections(output, frame.shape[:2])
            return detections

        except Exception as e:
            logger.error(f"Hailo inference error: {e}")
            return []

    def _parse_detections(self, output: dict, frame_shape: tuple) -> list[Detection]:
        """Parse model output into detections. Format depends on the specific model."""
        detections = []

        # Generic YOLO-style output parsing
        for name, data in output.items():
            if len(data.shape) >= 2:
                for detection in data[0]:
                    if len(detection) >= 6:
                        x, y, w, h, conf, class_id = detection[:6]

                        if conf < self.confidence_threshold:
                            continue

                        class_name = (
                            self._labels[int(class_id)]
                            if int(class_id) < len(self._labels)
                            else f"class_{int(class_id)}"
                        )

                        detections.append(Detection(
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h)),
                        ))

        return detections

    def cleanup(self):
        if self._vdevice:
            self._vdevice.release()
        self._network_group = None
        self._vdevice = None


class CPUBackend(InferenceBackend):
    """CPU-based inference using TensorFlow Lite."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._labels = []

    def load_model(self, model_path: str) -> bool:
        try:
            import tflite_runtime.interpreter as tflite

            self._interpreter = tflite.Interpreter(model_path=model_path)
            self._interpreter.allocate_tensors()

            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            # Load labels
            labels_path = Path(model_path).with_suffix(".txt")
            if labels_path.exists():
                self._labels = labels_path.read_text().strip().split("\n")

            logger.info(f"TFLite CPU model loaded: {model_path}")
            return True

        except ImportError:
            try:
                import tensorflow as tf

                self._interpreter = tf.lite.Interpreter(model_path=model_path)
                self._interpreter.allocate_tensors()

                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()

                labels_path = Path(model_path).with_suffix(".txt")
                if labels_path.exists():
                    self._labels = labels_path.read_text().strip().split("\n")

                logger.info(f"TFLite (TF) CPU model loaded: {model_path}")
                return True

            except ImportError:
                logger.error("Neither tflite_runtime nor tensorflow installed")
                return False

        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            return False

    def run_inference(self, frame: np.ndarray) -> list[Detection]:
        if self._interpreter is None:
            return []

        try:
            input_shape = self._input_details[0]["shape"]
            height, width = input_shape[1], input_shape[2]

            resized = cv2.resize(frame, (width, height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb, axis=0)

            if self._input_details[0]["dtype"] == np.uint8:
                input_data = input_data.astype(np.uint8)
            else:
                input_data = (input_data / 255.0).astype(np.float32)

            self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
            self._interpreter.invoke()

            # Get outputs (typical detection model format)
            boxes = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
            classes = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
            scores = self._interpreter.get_tensor(self._output_details[2]["index"])[0]

            detections = []
            for i, score in enumerate(scores):
                if score < self.confidence_threshold:
                    continue

                ymin, xmin, ymax, xmax = boxes[i]
                class_id = int(classes[i])

                class_name = (
                    self._labels[class_id] if class_id < len(self._labels) else f"class_{class_id}"
                )

                detections.append(Detection(
                    class_name=class_name,
                    confidence=float(score),
                    bbox=BoundingBox(
                        x=float(xmin),
                        y=float(ymin),
                        width=float(xmax - xmin),
                        height=float(ymax - ymin),
                    ),
                ))

            return detections

        except Exception as e:
            logger.error(f"CPU inference error: {e}")
            return []

    def cleanup(self):
        self._interpreter = None


class EdgeInference:
    """
    Edge inference manager supporting multiple accelerator backends.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._backend: Optional[InferenceBackend] = None
        self._last_inference_time: float = 0
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the inference backend."""
        if not self.config.enabled:
            logger.info("Edge inference disabled")
            return False

        # Select backend
        if self.config.accelerator == "coral":
            self._backend = CoralBackend(self.config.confidence_threshold)
        elif self.config.accelerator == "hailo":
            self._backend = HailoBackend(self.config.confidence_threshold)
        else:
            self._backend = CPUBackend(self.config.confidence_threshold)

        # Load model
        if not Path(self.config.model_path).exists():
            logger.error(f"Model not found: {self.config.model_path}")
            return False

        if self._backend.load_model(self.config.model_path):
            self._initialized = True
            logger.info(f"Edge inference initialized with {self.config.accelerator} backend")
            return True

        return False

    def run(self, frame: np.ndarray, timestamp: float) -> Optional[list[Detection]]:
        """
        Run inference on a frame if enough time has passed.

        Returns:
            List of detections or None if skipped
        """
        if not self._initialized or not self._backend:
            return None

        # Rate limiting
        time_since_last = (timestamp - self._last_inference_time) * 1000
        if time_since_last < self.config.inference_interval_ms:
            return None

        self._last_inference_time = timestamp

        start = time.time()
        detections = self._backend.run_inference(frame)
        elapsed_ms = (time.time() - start) * 1000

        if detections:
            logger.debug(f"Edge inference: {len(detections)} detections in {elapsed_ms:.1f}ms")

        return detections

    def cleanup(self):
        """Cleanup inference backend."""
        if self._backend:
            self._backend.cleanup()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

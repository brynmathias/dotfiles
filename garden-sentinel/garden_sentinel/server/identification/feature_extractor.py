"""
Feature extraction for predator identification.

Extracts visual features that can be used to identify individual animals.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """A feature vector representing an animal's appearance."""
    vector: List[float]
    confidence: float = 1.0
    source_frame: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    timestamp: float = 0.0

    def dimension(self) -> int:
        """Get vector dimension."""
        return len(self.vector)

    def normalize(self) -> "FeatureVector":
        """Return L2-normalized vector."""
        magnitude = math.sqrt(sum(x * x for x in self.vector))
        if magnitude < 1e-10:
            return self
        normalized = [x / magnitude for x in self.vector]
        return FeatureVector(
            vector=normalized,
            confidence=self.confidence,
            source_frame=self.source_frame,
            bbox=self.bbox,
            timestamp=self.timestamp,
        )

    def cosine_similarity(self, other: "FeatureVector") -> float:
        """Calculate cosine similarity with another vector."""
        if len(self.vector) != len(other.vector):
            raise ValueError("Vector dimensions must match")

        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))
        mag_a = math.sqrt(sum(x * x for x in self.vector))
        mag_b = math.sqrt(sum(x * x for x in other.vector))

        if mag_a < 1e-10 or mag_b < 1e-10:
            return 0.0

        return dot_product / (mag_a * mag_b)

    def euclidean_distance(self, other: "FeatureVector") -> float:
        """Calculate Euclidean distance to another vector."""
        if len(self.vector) != len(other.vector):
            raise ValueError("Vector dimensions must match")

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.vector, other.vector)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector,
            "confidence": self.confidence,
            "source_frame": self.source_frame,
            "bbox": self.bbox,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureVector":
        return cls(
            vector=data["vector"],
            confidence=data.get("confidence", 1.0),
            source_frame=data.get("source_frame"),
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            timestamp=data.get("timestamp", 0.0),
        )


class FeatureExtractor:
    """
    Extracts identifying features from animal images.

    Uses a pre-trained model to generate feature embeddings
    that can be used for re-identification.
    """

    EMBEDDING_DIM = 128  # Output dimension

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_gpu: bool = False,
    ):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._model = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the feature extraction model."""
        try:
            if self.model_path and self.model_path.exists():
                self._load_model()
            else:
                # Use lightweight fallback
                self._init_fallback()

            self._initialized = True
            logger.info("Feature extractor initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize feature extractor: {e}")
            return False

    def _load_model(self):
        """Load the pre-trained model."""
        try:
            import torch
            import torchvision.models as models

            # Load a ResNet backbone for feature extraction
            self._model = models.resnet18(pretrained=False)
            self._model.fc = torch.nn.Identity()  # Remove classification head

            if self.model_path:
                state_dict = torch.load(self.model_path, map_location="cpu")
                self._model.load_state_dict(state_dict, strict=False)

            self._model.eval()

            if self.use_gpu and torch.cuda.is_available():
                self._model = self._model.cuda()

            logger.info("Loaded PyTorch model for feature extraction")

        except ImportError:
            logger.warning("PyTorch not available, using fallback")
            self._init_fallback()

    def _init_fallback(self):
        """Initialize fallback feature extraction (color histograms, etc.)."""
        self._model = "fallback"
        logger.info("Using fallback feature extraction")

    def extract(
        self,
        image_data: bytes,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        predator_type: Optional[str] = None,
    ) -> Optional[FeatureVector]:
        """
        Extract features from an image.

        Args:
            image_data: Raw image bytes (JPEG/PNG)
            bbox: Optional bounding box (x1, y1, x2, y2) to crop
            predator_type: Type of predator (for specialized processing)

        Returns:
            FeatureVector or None if extraction fails
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            if self._model == "fallback":
                return self._extract_fallback(image_data, bbox)
            else:
                return self._extract_neural(image_data, bbox)

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def _extract_neural(
        self,
        image_data: bytes,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> FeatureVector:
        """Extract features using neural network."""
        import torch
        from PIL import Image
        import io

        # Load and preprocess image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        if bbox:
            image = image.crop(bbox)

        # Resize to model input size
        image = image.resize((224, 224))

        # Convert to tensor
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        tensor = transform(image).unsqueeze(0)

        if self.use_gpu and torch.cuda.is_available():
            tensor = tensor.cuda()

        # Extract features
        with torch.no_grad():
            features = self._model(tensor)

        # Convert to list
        vector = features.squeeze().cpu().numpy().tolist()

        return FeatureVector(
            vector=vector[:self.EMBEDDING_DIM],  # Truncate if needed
            confidence=1.0,
            bbox=bbox,
        )

    def _extract_fallback(
        self,
        image_data: bytes,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> FeatureVector:
        """
        Extract features using traditional CV methods.

        Uses color histograms and basic shape features.
        """
        try:
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            if bbox:
                image = image.crop(bbox)

            # Resize for consistent features
            image = image.resize((64, 64))

            # Extract color histogram
            pixels = list(image.getdata())

            # RGB histograms (32 bins each = 96 features)
            r_hist = [0] * 32
            g_hist = [0] * 32
            b_hist = [0] * 32

            for r, g, b in pixels:
                r_hist[r // 8] += 1
                g_hist[g // 8] += 1
                b_hist[b // 8] += 1

            # Normalize histograms
            total = len(pixels)
            r_hist = [x / total for x in r_hist]
            g_hist = [x / total for x in g_hist]
            b_hist = [x / total for x in b_hist]

            # Extract edge features (simplified)
            # Use gradient magnitude at key points
            edge_features = self._extract_edge_features(image)

            # Combine features
            vector = r_hist + g_hist + b_hist + edge_features

            # Pad or truncate to EMBEDDING_DIM
            if len(vector) < self.EMBEDDING_DIM:
                vector.extend([0.0] * (self.EMBEDDING_DIM - len(vector)))
            else:
                vector = vector[:self.EMBEDDING_DIM]

            return FeatureVector(
                vector=vector,
                confidence=0.8,  # Lower confidence for fallback
                bbox=bbox,
            )

        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            # Return zero vector
            return FeatureVector(
                vector=[0.0] * self.EMBEDDING_DIM,
                confidence=0.0,
            )

    def _extract_edge_features(self, image) -> List[float]:
        """Extract basic edge/gradient features."""
        # Simplified Sobel-like gradient features
        pixels = list(image.getdata())
        width, height = image.size

        def get_pixel(x, y):
            if 0 <= x < width and 0 <= y < height:
                idx = y * width + x
                r, g, b = pixels[idx]
                return (r + g + b) / 3 / 255
            return 0

        features = []
        # Sample gradients at grid points
        for gy in range(4):
            for gx in range(4):
                x = (gx + 1) * width // 5
                y = (gy + 1) * height // 5

                # Horizontal gradient
                gx_val = get_pixel(x + 1, y) - get_pixel(x - 1, y)
                # Vertical gradient
                gy_val = get_pixel(x, y + 1) - get_pixel(x, y - 1)
                # Magnitude
                mag = math.sqrt(gx_val ** 2 + gy_val ** 2)

                features.append(mag)

        return features

    def compute_similarity(
        self,
        feature1: FeatureVector,
        feature2: FeatureVector,
    ) -> float:
        """
        Compute similarity between two feature vectors.

        Returns a score between 0 and 1.
        """
        # Use cosine similarity
        sim = feature1.cosine_similarity(feature2)

        # Map from [-1, 1] to [0, 1]
        return (sim + 1) / 2

    def batch_extract(
        self,
        images: List[Tuple[bytes, Optional[Tuple[int, int, int, int]]]],
    ) -> List[Optional[FeatureVector]]:
        """
        Extract features from multiple images.

        More efficient than calling extract() repeatedly.
        """
        return [self.extract(img, bbox) for img, bbox in images]

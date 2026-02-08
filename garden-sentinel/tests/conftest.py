"""
Shared test fixtures for Garden Sentinel tests.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_garden_config():
    """Sample garden configuration for testing."""
    return {
        "name": "Test Garden",
        "boundary": [
            [0, 0],
            [100, 0],
            [100, 50],
            [0, 50],
        ],
        "zones": [
            {
                "id": "protected_zone",
                "name": "Chicken Run",
                "type": "protected",
                "vertices": [
                    [20, 10],
                    [80, 10],
                    [80, 40],
                    [20, 40],
                ],
            },
            {
                "id": "entry_north",
                "name": "North Entry",
                "type": "entry_point",
                "vertices": [
                    [40, 48],
                    [60, 48],
                    [60, 50],
                    [40, 50],
                ],
            },
        ],
        "cameras": [
            {
                "id": "cam-01",
                "name": "Front Camera",
                "x": 50,
                "y": 0,
                "heading": 0,
                "fov": 90,
                "range": 30,
            },
        ],
        "gps_origin": {
            "latitude": 51.5074,
            "longitude": -0.1278,
            "altitude": 0,
        },
    }


@pytest.fixture
def sample_detections():
    """Sample detection results for testing."""
    return [
        {
            "class_name": "fox",
            "confidence": 0.92,
            "bbox": {"x": 100, "y": 150, "width": 80, "height": 60},
            "track_id": "track_001",
        },
        {
            "class_name": "cat",
            "confidence": 0.78,
            "bbox": {"x": 300, "y": 200, "width": 50, "height": 40},
            "track_id": "track_002",
        },
    ]

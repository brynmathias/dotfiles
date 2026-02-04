# Garden Sentinel 🐔

AI-powered garden security system to protect your chickens from predators.

## Features

- **Multi-camera support**: Deploy multiple Raspberry Pi 5 cameras around your garden
- **Predator detection**: YOLO-based detection for foxes, badgers, cats, dogs, birds of prey, and more
- **Automated responses**: Alarms and water sprayer activation based on threat level
- **Edge inference**: Optional on-device detection using Google Coral TPU or Hailo accelerators
- **Object tracking**: Track detected predators for precise targeting
- **Servo control**: Pan/tilt camera mounts and water gun aiming
- **Fine-tuning**: Train custom models with your own images
- **Real-time dashboard**: Web-based monitoring and control

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Central Server                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │  Detection  │  │   Alert     │  │   Storage   │  │   Dashboard    │  │
│  │  Pipeline   │  │   Manager   │  │   Manager   │  │   (FastAPI)    │  │
│  │  (YOLO)     │  │             │  │  (SQLite)   │  │                │  │
│  └──────▲──────┘  └──────┬──────┘  └─────────────┘  └────────────────┘  │
│         │                │                                               │
│         │           MQTT │ Commands                                      │
└─────────┼────────────────┼───────────────────────────────────────────────┘
          │                │
    Frames│          ┌─────▼─────┐
          │          │   MQTT    │
          │          │  Broker   │
          │          └─────┬─────┘
          │                │
    ┌─────┴────────────────┼────────────────────────────────────┐
    │                      │                                     │
┌───▼───┐            ┌─────▼─────┐                         ┌─────▼─────┐
│ Edge  │            │   Edge    │         ...             │   Edge    │
│ Cam 1 │            │   Cam 2   │                         │   Cam N   │
│       │            │           │                         │           │
│ Pi 5  │            │   Pi 5    │                         │   Pi 5    │
│+Coral │            │  +Hailo   │                         │           │
└───┬───┘            └─────┬─────┘                         └─────┬─────┘
    │                      │                                     │
    ▼                      ▼                                     ▼
[Camera]              [Camera]                              [Camera]
[Servos]              [Servos]                              [Servos]
[Alarm]               [Sprayer]                             [Alarm]
```

## Installation

### Using UV (recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourrepo/garden-sentinel.git
cd garden-sentinel

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate

# Install for server (with GPU support)
uv pip install -e ".[server]"

# Install for edge device (Raspberry Pi)
uv pip install -e ".[edge]"

# Install with Coral TPU support
uv pip install -e ".[edge,coral]"

# Install everything for development
uv pip install -e ".[server,edge,coral,training,dev]"
```

### Hardware Requirements

**Server:**
- Any machine with Python 3.11+
- NVIDIA GPU recommended for faster inference
- Or Intel/AMD CPU for smaller models

**Edge Devices (Raspberry Pi 5):**
- Raspberry Pi 5 (4GB+ RAM recommended)
- Raspberry Pi Camera Module 3 or compatible USB camera
- Optional: Google Coral USB Accelerator or Hailo-8 M.2 module
- Optional: Servo motors for pan/tilt mount
- Optional: Relay module for alarm/sprayer control

## Quick Start

### 1. Start MQTT Broker

The system uses MQTT for real-time communication. Install and start Mosquitto:

```bash
# Ubuntu/Debian
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto

# macOS
brew install mosquitto
brew services start mosquitto
```

### 2. Start the Server

```bash
# Copy and edit the configuration
cp server/config/config.yaml my-config.yaml
# Edit my-config.yaml with your settings

# Start the server
garden-sentinel-server -c my-config.yaml

# Or run directly
python -m garden_sentinel.server.main -c my-config.yaml
```

The server will be available at:
- Dashboard: http://localhost:5000
- API: http://localhost:5000/api
- WebSocket: ws://localhost:5000/ws

### 3. Start Edge Devices

On each Raspberry Pi:

```bash
# Copy and edit the configuration
cp edge/config/config.yaml /etc/garden-sentinel/edge.yaml
# Edit with your device settings

# Start the edge device
garden-sentinel-edge -c /etc/garden-sentinel/edge.yaml
```

## Configuration

### Server Configuration (`server/config/config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 5000

mqtt:
  enabled: true
  broker: "localhost"
  port: 1883

detection:
  model_type: "yolov8"
  use_pretrained: true  # Use COCO-pretrained model initially
  pretrained_model: "yolov8n.pt"  # n=nano, s=small, m=medium, l=large, x=xlarge
  confidence_threshold: 0.5
  device: "cuda"  # or "cpu"

alerts:
  cooldown_s: 30
  critical:
    notify: true
    alarm: true
    sprayer: true
```

### Edge Configuration (`edge/config/config.yaml`)

```yaml
device:
  id: "garden-cam-01"
  name: "Chicken Coop Camera"
  location: "coop_entrance"

camera:
  width: 1920
  height: 1080
  fps: 30

streaming:
  http_enabled: true
  http_port: 8080

server:
  host: "192.168.1.100"
  port: 5000
  mqtt_broker: "192.168.1.100"

gpio:
  alarm_pin: 17
  sprayer_pin: 27

edge_inference:
  enabled: true
  accelerator: "coral"  # or "hailo", "cpu"
```

## Training Custom Models

The system works out-of-the-box with COCO-pretrained models, but you can fine-tune for better predator detection:

### 1. Collect Training Data

The system automatically saves detected frames. You can also import external datasets:

```bash
# Import from system events
garden-sentinel-collect import-events --events-dir data/events

# Import external dataset (e.g., from Roboflow)
garden-sentinel-collect import-external --images path/to/images --labels path/to/labels

# View statistics
garden-sentinel-collect stats

# Run interactive labeling tool
garden-sentinel-collect label
```

### 2. Prepare Dataset

```bash
garden-sentinel-train prepare \
  --images data/training/images \
  --labels data/training/labels \
  --val-split 0.2
```

### 3. Train Model

```bash
garden-sentinel-train train \
  --dataset data/dataset/dataset.yaml \
  --epochs 100 \
  --batch-size 16 \
  --device 0  # GPU 0
```

### 4. Export for Edge Devices

```bash
# Export for Coral TPU
garden-sentinel-train export --model models/best.pt --coral

# Export to ONNX
garden-sentinel-train export --model models/best.pt --format onnx
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/frames` | POST | Upload frame from edge device |
| `/api/devices` | GET | List all registered devices |
| `/api/devices/{id}/command` | POST | Send command to device |
| `/api/events` | GET | Get alert events |
| `/api/events/{id}/frame` | GET | Get event frame |
| `/api/stats` | GET | Get system statistics |
| `/api/training/samples` | GET | Get training samples |
| `/ws` | WebSocket | Real-time updates |

## Supported Predators

The system recognizes these predator classes:

| Class | Threat Level | Default Action |
|-------|--------------|----------------|
| Fox | Critical | Alarm + Sprayer |
| Weasel | Critical | Alarm + Sprayer |
| Stoat | Critical | Alarm + Sprayer |
| Mink | Critical | Alarm + Sprayer |
| Hawk | Critical | Alarm + Sprayer |
| Eagle | Critical | Alarm + Sprayer |
| Badger | High | Alarm |
| Owl | High | Alarm |
| Dog | High | Alarm |
| Cat | Medium | Alert |
| Crow | Medium | Alert |
| Magpie | Medium | Alert |
| Rat | Medium | Alert |

## Hardware Wiring

### GPIO Pin Assignments (BCM numbering)

| Pin | Function | Notes |
|-----|----------|-------|
| 17 | Alarm relay | Connect to relay IN |
| 27 | Sprayer relay | Connect to relay IN |
| 22 | Status LED | Optional indicator |
| 23 | IR LED control | For night vision |
| 12 | Camera pan servo | PWM |
| 13 | Camera tilt servo | PWM |
| 18 | Gun pan servo | PWM |
| 19 | Gun tilt servo | PWM |

### Servo Connections

```
Servo Signal -> GPIO Pin
Servo Power  -> 5V (external power recommended for servos)
Servo Ground -> Common ground
```

## Troubleshooting

### Camera not detected
```bash
# Check if camera is recognized
libcamera-hello
# Or for older systems
raspistill -o test.jpg
```

### Coral TPU not found
```bash
# Install Coral runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std python3-pycoral
```

### MQTT connection issues
```bash
# Test MQTT broker
mosquitto_pub -t test -m "hello"
mosquitto_sub -t test
```

## License

MIT License - see LICENSE file for details.

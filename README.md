# Snake Hand Control

An interactive Snake game controlled by hand gestures using computer vision.

## Getting Started

### Prerequisites

- Python 3.7+
- OpenCV
- Pygame
- MediaPipe

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MarineRvdy/snake_hand_control.git
   cd snake_hand_control
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### How to Play

1. Run the game:
   ```bash
   python src/main.py
   ```

2. Hand Gesture Controls:

   **Right Hand (Movement Control)**
   - Point your index finger to control the snake's direction:
     - Point up: Move snake up
     - Point down: Move snake down
     - Point left: Move snake left
     - Point right: Move snake right
   - The direction is determined by the position of your index finger relative to your wrist

   **Left Hand (Speed Control)**
   - Make a fist: Increase snake speed
   - Open hand: Decrease snake speed
   - Keep your hand flat for normal speed

   **Special Controls**
   - Make de the "OK" sign (touch thumb and index finger together): Restart game or continue game after pause
   - Make the "dead time" sign: Pause game


## 📁 Project Structure

- `src/` - Main source code
  - `Game.py` - Main game logic and rendering
  - `HandController.py` - Hand tracking and gesture recognition
  - `Snake.py` - Snake game mechanics
  - `main.py` - Entry point
- `first_versions/` - Early development versions


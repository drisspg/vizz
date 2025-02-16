# Vizz

Collection of math visualizations using Manim.

## Project Structure

```
vizz/
├── flex/       # Animations for FlexAttention
└── quant/      # Quant Animations
```

## Installation

1. Install the package in development mode:
```bash
pip install -e .
```

### macOS Dependencies
This won't be installed by default,

You can install the required system dependencies using Homebrew:

```bash
brew install ffmpeg mactex
```
## Usage

### Production Renders

Run animations using ManimGL for high-quality output:

```bash
manimgl file.py SceneName
```

Example:
```bash
manimgl vizz/flex/end_to_end.py AttentionScoresVisualization
```

### Interactive Development

ManimGL provides a powerful interactive development workflow:

1. Start the scene in interactive mode:
```bash
manimgl file.py SceneName -p
```

<!-- 2. Use the interactive interface:
- Press 'q' to quit
- Press 'r' to refresh scene
- Use arrow keys to move through animations
- Press 'space' to play/pause
- Press '[' and ']' to move through frames
- Press 'scroll lock' to enter scroll mode
  - In scroll mode, use the mouse to navigate the scene
  - Scroll wheel to zoom in/out
  - Right click and drag to move camera
  - Left click and drag to rotate view

This interactive mode is especially useful when:
- Tweaking animation timings
- Adjusting object positions
- Fine-tuning camera angles
- Debugging complex animations
- Testing scene transitions -->

## Requirements

- Python 3.x
- ManimGL

## Development Tips

1. Start with interactive mode (-p flag) during development
2. Use the refresh feature (r key) to see changes without restarting
3. Save final renders with high quality settings once satisfied
4. Consider using the -l flag for low quality during initial development

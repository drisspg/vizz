# Vizz

Collection of math visualizations using Manim.

## Project Structure

```
vizz/
├── flex/       # Animations for FlexAttention
└── quant/      # Quant Animations
```

## Usage

Run animations using Manim:

```bash
# High quality render
manim -pqh file.py SceneName

# Medium quality render
manim -pqm file.py SceneName

# Low quality render
manim -pql file.py SceneName
```

Example:
```bash
manim -pqh flex/end_to_end.py AttentionScoresVisualization
```

## Requirements

- Python 3.x
- Manim
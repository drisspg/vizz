# Vizz

Collection of math visualizations using Manim and Manim-Slides for interactive presentations.

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

2. Install manim-slides for presentation functionality:
```bash
pip install manim-slides
```

### macOS Dependencies

You can install the required system dependencies using Homebrew:

```bash
brew install ffmpeg mactex
```

## Usage

This project uses both **ManimGL** for animations and **manim-slides** for interactive presentations.

### Running Animations

#### Basic Animation Rendering

For standard animations without slide functionality:

```bash
manimgl file.py SceneName
```

Example:
```bash
manimgl vizz/flex/end_to_end.py AttentionScoresVisualization
```

#### Creating Interactive Slides

For slide-based presentations (classes that inherit from `Slide`):

```bash
manim-slides render file.py SceneName
```

Examples:
```bash
# Render NATTEN visualization slides
manim-slides render vizz/flex/natten.py RasterizationComparison

# Render with custom environment variables
ORDER=morton manim-slides render vizz/flex/natten.py RasterizationComparison

# Render score modification slides
manim-slides render vizz/flex/score_mod.py ScoreModAttentionVisualization
```

#### Presenting Slides

After rendering slides, start the presentation:

```bash
manim-slides present SceneName
```

Example:
```bash
manim-slides present RasterizationComparison
```

**Presentation Controls:**
- `Space` or `Right Arrow`: Next slide
- `Left Arrow`: Previous slide
- `R`: Restart presentation
- `Q` or `Esc`: Quit presentation

### Quality Settings

#### Low Quality (Fast Preview)
```bash
# For ManimGL
manimgl file.py SceneName -ql

# For manim-slides
manim-slides render file.py SceneName -ql
```

#### High Quality (Final Output)
```bash
# For ManimGL
manimgl file.py SceneName -qh

# For manim-slides
manim-slides render file.py SceneName -qh
```

### Interactive Development

ManimGL provides a powerful interactive development workflow:

```bash
manimgl file.py SceneName -p
```

### Available Animations

#### FlexAttention (`vizz/flex/`)

| File | Scene Class | Description | Type |
|------|-------------|-------------|------|
| `end_to_end.py` | `AttentionScoresVisualization` | Complete attention mechanism walkthrough | Animation |
| `natten.py` | `NattenBasicVisualization` | NATTEN neighborhood attention basics | Slide |
| `natten.py` | `RasterizationComparison` | Compare rasterization patterns | Slide |
| `score_mod.py` | `ScoreModAttentionVisualization` | Score modification functions | Slide |
| `mod_scene.py` | `MaskAnimationScene` | Attention mask visualization | Slide |
| `causal_attention.py` | `CausalAttentionVisualization` | Causal attention masking | Slide |
| `block_mask.py` | `BlockMaskKVCreation` | Block mask construction | Slide |

#### Examples with Environment Variables

```bash
# NATTEN with Morton order
ORDER=morton manim-slides render vizz/flex/natten.py RasterizationComparison

# NATTEN with row-major order (default)
ORDER=row_major manim-slides render vizz/flex/natten.py RasterizationComparison
```

### Output Files

- **Animations**: Video files saved to `videos/` directory
- **Slides**: HTML presentations saved for interactive viewing
- **Images**: Individual frames saved as PNG files

## Development Tips

1. **Start with low quality** (`-ql`) during development for faster iteration
2. **Use interactive mode** (`-p`) with ManimGL for real-time adjustments
3. **Test slides** with `manim-slides render` before final presentation
4. **Environment variables** can customize animation behavior (see individual files)
5. **Preview slides** with `manim-slides present` to test navigation

## Requirements

- Python 3.7+
- ManimGL
- manim-slides
- PyTorch
- PIL (Pillow)
- NumPy

## Troubleshooting

### Common Issues

1. **Module not found**: Ensure you've installed with `pip install -e .`
2. **ffmpeg missing**: Install with `brew install ffmpeg` on macOS
3. **LaTeX errors**: Install with `brew install mactex` on macOS
4. **Slides won't present**: Make sure you rendered them first with `manim-slides render`

### Getting Help

Check individual animation files for specific usage instructions - many contain command examples in their docstrings.

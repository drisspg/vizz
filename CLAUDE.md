# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vizz is a visualization library for creating mathematical and attention mechanism animations using ManimGL and manim-slides. The project focuses on visualizing FlexAttention patterns, quantization, and various attention mechanisms for educational and presentation purposes.

## Development Environment

**Required:** Always activate the manim conda environment before running any commands:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate manim
```

## Common Development Commands

### Installation
```bash
# Install the package in development mode (required for imports to work)
pip install -e .
```

### Running Animations

#### For standard animations (non-slide classes):
```bash
manimgl vizz/flex/[file].py [SceneName]
# Example:
manimgl vizz/flex/end_to_end.py AttentionScoresVisualization
```

#### For slide-based presentations (classes inheriting from Slide):
```bash
# Render slides
manim-slides render vizz/flex/[file].py [SceneName]

# Present slides after rendering
manim-slides present [SceneName]
```

#### Quality options:
- `-ql` for low quality (fast preview)
- `-qh` for high quality (final output)
- `-p` for interactive development mode (ManimGL only)

### Code Quality

```bash
# Run linting
ruff check vizz/

# Run linting with auto-fix
ruff check --fix vizz/

# Format check
ruff format --check vizz/

# Format code
ruff format vizz/
```

## Project Architecture

### Directory Structure
```
vizz/
├── flex/              # FlexAttention animations
│   ├── block_mask.py         # Block mask construction visualization
│   ├── causal_attention.py   # Causal attention masking
│   ├── end_to_end.py         # Complete attention mechanism walkthrough
│   ├── mod_scene.py          # Attention mask visualization
│   ├── natten.py            # NATTEN neighborhood attention
│   ├── ordering_comparison.py # Different ordering patterns comparison
│   └── score_mod.py          # Score modification functions
└── quant/            # Quantization animations
    └── scaling.py    # Scaling visualizations
```

### Key Classes and Patterns

1. **Slide Classes**: Classes that inherit from `Slide` are interactive presentations
   - Must be rendered with `manim-slides render`
   - Support navigation controls during presentation

2. **Animation Classes**: Standard Manim scenes for video output
   - Run with `manimgl` command
   - Can use interactive mode with `-p` flag

3. **Common Imports**:
   ```python
   from manim import *
   from manim_slides import Slide
   from manim import config
   ```

4. **Configuration**: Light theme is used by default:
   ```python
   config.background_color = WHITE
   ```

### Environment Variables

Some animations support environment variables for customization:
```bash
# Example: Change rasterization order in NATTEN
ORDER=morton manim-slides render vizz/flex/natten.py RasterizationComparison
ORDER=row_major manim-slides render vizz/flex/natten.py RasterizationComparison
```

## Available Animation Scenes

### Slide-based Presentations
- `ScoreModAttentionVisualization` - Score modification functions
- `NattenBasicVisualization` - NATTEN neighborhood attention basics
- `RasterizationComparison` - Compare rasterization patterns
- `MaskAnimationScene` - Attention mask visualization
- `BlockMaskKVCreation` - Block mask construction
- `AttentionScoresVisualization` - Complete attention mechanism
- `OrderingPatterns` - Different ordering patterns comparison
- `CausalAttentionVisualization` - Causal attention masking

## Output Locations

- **Videos**: `videos/[SceneName]/`
- **Slides**: `slides/` (JSON presentation data)
- **Media assets**: `media/` (generated images, SVGs, etc.)

## Testing Animations

When modifying animations:
1. First test with low quality: `manimgl file.py SceneName -ql`
2. Use interactive mode for development: `manimgl file.py SceneName -p`
3. For slides, render and test navigation: `manim-slides render file.py SceneName -ql && manim-slides present SceneName`
4. Final render in high quality: `manim-slides render file.py SceneName -qh`

## Dependencies

Core dependencies (from pyproject.toml):
- `torch` - PyTorch for tensor operations
- `manimgl==1.7.2` - ManimGL for animations
- `manim-slides` - For interactive presentations
- `ruff` - Linting and formatting
- `attn_gym` - Attention mechanisms library (external dependency)

## Important Notes

1. Always ensure the manim conda environment is activated
2. The project must be installed with `pip install -e .` for imports to work
3. Slide classes require `manim-slides` commands, not `manimgl`
4. Interactive development mode (`-p`) only works with `manimgl`, not `manim-slides`
5. The project uses a light theme (white background) for presentations

"""Shared utilities and base classes for MX/NVFP4 quantization visualizations."""

from manim import *
from manim_slides import Slide

# Light theme configuration
config.background_color = WHITE

# Color scheme
COLORS = {
    "text": BLACK,
    "matrix": DARK_GRAY,
    "block_boundary": GRAY,
    "active_block": YELLOW_E,
    "computed_scale": GREEN_D,
    "max_value": RED_D,
    "tensor_data": BLUE_D,
    "mx_color": PURPLE_D,
    "nvfp_color": ORANGE,
    "packed_data": TEAL_D,
    "bf16_color": BLUE_C,
    "fp8_color": ORANGE,
}

# Layout configuration
LAYOUT = {
    "block_width": 1.2,
    "block_height": 0.35,
    "block_spacing": 0.08,
    "scale_cell_width": 0.55,
    "scale_cell_height": 0.4,
    "top_y": 1.3,
    "bottom_y": -2.0,
}

# Display configuration
# Set to True to show full matrix grid with all 32 values (8x4 layout)
# Set to False to show compact text representation (e.g., "val1 val2 val3...")
#
# Example:
#   SHOW_FULL_MATRICES = False  →  "1.2 3.4 5.6..."
#   SHOW_FULL_MATRICES = True   →   1.2   3.4   5.6   7.8
#                                    2.1   4.3   6.5   8.7
#                                    ...   ...   ...   ...
#
# Note: Full matrix mode uses Manim's MobjectMatrix for proper grid layout rendering.
SHOW_FULL_MATRICES = True


class BlockVisualizationBase(Slide):
    """Base class with helper methods for block visualization."""

    def create_block_grid(
        self,
        num_rows,
        num_blocks,
        start_x,
        start_y,
        width,
        height,
        spacing,
        color,
        show_values=None,
        get_matrix_values=None,
    ):
        """Create a grid of blocks with optional value labels.

        Args:
            num_rows: Number of rows in the grid
            num_blocks: Number of blocks per row
            start_x: Starting x position
            start_y: Starting y position
            width: Width of each block
            height: Height of each block
            spacing: Spacing between blocks
            color: Color of the blocks
            show_values: Optional function(row, col) -> str to display values (compact mode)
            get_matrix_values: Optional function(row, col) -> numpy array for Matrix display

        Returns:
            VGroup of blocks and list of (rect, label) tuples for each row
        """
        blocks_group = VGroup()
        block_objects = []

        for row in range(num_rows):
            row_blocks = []
            for col in range(num_blocks):
                # Create block rectangle
                block_rect = Rectangle(
                    width=width,
                    height=height,
                    color=color,
                    stroke_width=2,
                    fill_opacity=0.15,
                    fill_color=color,
                )

                # Position the block
                x_pos = start_x + col * (width + spacing) + width / 2
                y_pos = start_y - row * (height + 0.12)
                block_rect.move_to([x_pos, y_pos, 0])

                # Create label based on display mode
                if SHOW_FULL_MATRICES and get_matrix_values:
                    # Show full matrix using efficient text grid (not Matrix object)
                    matrix_vals = get_matrix_values(row, col)
                    value_label = self._create_text_matrix(matrix_vals, block_rect)
                elif show_values:
                    # Show compact text
                    value_text = show_values(row, col)
                    value_label = Text(value_text, font_size=8, color=COLORS["text"])
                    value_label.move_to(block_rect.get_center())
                else:
                    # Show placeholder
                    value_label = Text("?", font_size=12, color=COLORS["text"])
                    value_label.move_to(block_rect.get_center())

                blocks_group.add(VGroup(block_rect, value_label))
                row_blocks.append((block_rect, value_label))
            block_objects.append(row_blocks)

        return blocks_group, block_objects

    def create_title(self, text, font_size=36, color=None, position=None):
        """Create a title text object."""
        color = color or COLORS["text"]
        title = Text(text, font_size=font_size, color=color, weight=BOLD)
        if position is not None:
            title.move_to(position)
        return title

    def _create_text_matrix(self, matrix_vals, block_rect):
        """Create a matrix display using Manim's MobjectMatrix with Text objects.

        Args:
            matrix_vals: 2D numpy array of numeric values (e.g., 8x4 array)
            block_rect: The rectangle to center the matrix in

        Returns:
            MobjectMatrix object centered in the block
        """
        # Convert numeric values to Text objects for each cell
        text_matrix = []
        for row_vals in matrix_vals:
            text_row = []
            for val in row_vals:
                text_obj = Text(f"{val:.1f}", font_size=6, color=COLORS["text"])
                text_row.append(text_obj)
            text_matrix.append(text_row)

        # Create MobjectMatrix without brackets
        matrix_mob = MobjectMatrix(
            text_matrix,
            v_buff=0.1,  # Very tight vertical spacing
            h_buff=0.15,  # Very tight horizontal spacing
        )

        # Remove brackets
        matrix_mob.get_brackets().set_opacity(0)

        # Scale to fit within block while maintaining aspect ratio
        max_width = block_rect.width * 0.95
        max_height = block_rect.height * 0.9

        if matrix_mob.width > max_width:
            matrix_mob.scale(max_width / matrix_mob.width)
        if matrix_mob.height > max_height:
            matrix_mob.scale(max_height / matrix_mob.height)

        # Center in block
        matrix_mob.move_to(block_rect.get_center())
        return matrix_mob

    def animate_block_fill(
        self, rect, label, new_text, new_color, opacity=0.25, run_time=0.25
    ):
        """Animate filling a block with new value and color."""
        new_label = Text(new_text, font_size=8, color=COLORS["text"])
        new_label.move_to(label.get_center())

        self.play(
            rect.animate.set_fill(new_color, opacity=opacity),
            Transform(label, new_label),
            run_time=run_time,
        )

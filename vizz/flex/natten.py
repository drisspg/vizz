"""
To run call: manimgl natten_visualization.py RasterizationComparison
"""

import torch
import numpy as np
from manim import *
from manim_slides import Slide
from manim import config
from PIL import Image
from attn_gym.masks.natten import morton_encode
import os

# Custom configuration to use a light theme for presentations
config.background_color = WHITE

# Constants for consistent styling
COLORS = {
    "text": BLACK,
    "matrix": DARK_GRAY,
    "bracket": DARK_GRAY,
    "highlight": {
        "query": GOLD_D,
        "key": BLUE_D,
        "result": GREEN_D,
        "masked": RED_D,
        "neighborhood": PURPLE_D,
        "window": RED_D,
        "path": GREEN_E,
        "morton": TEAL_D,
    },
    "pixel_grid": LIGHT_GRAY,
}

OPACITY = {"highlight": 0.2}


class PixelGrid:
    """Helper class for creating and manipulating pixel grids"""

    def __init__(self, grid_size, pixel_size=0.5, with_values=True):
        self.grid_size = grid_size
        self.pixel_size = pixel_size
        self.with_values = with_values
        self.grid = None
        self.pixel_map = {}  # Maps (row, col) to pixel mobjects

    def create_grid(self, data_tensor=None):
        """Create a visual grid representation of data"""
        grid = VGroup()

        # If no data provided, create a gradient pattern
        if data_tensor is None:
            data_tensor = torch.zeros((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Create a pattern with values between 0-255
                    value = ((j / self.grid_size) * 128) + ((i / self.grid_size) * 128)
                    data_tensor[i, j] = value

            # Process as grayscale
            is_color = False
        else:
            # Check if the tensor has color channels
            is_color = len(data_tensor.shape) > 2 and data_tensor.shape[2] >= 3

        # Create grid of squares
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Determine pixel color based on whether it's color or grayscale
                if is_color:
                    # For color images, use the RGB values directly
                    r = data_tensor[i, j, 0].item() / 255
                    g = data_tensor[i, j, 1].item() / 255
                    b = data_tensor[i, j, 2].item() / 255
                    pixel_color = rgb_to_color([r, g, b])
                    brightness = (
                        0.299 * r + 0.587 * g + 0.114 * b
                    )  # For text color determination
                else:
                    # For grayscale, normalize and use the same value for R, G, B
                    if torch.max(data_tensor) > 0:
                        pixel_value = (
                            data_tensor[i, j].item() / torch.max(data_tensor).item()
                        )
                    else:
                        pixel_value = 0
                    color_value = pixel_value
                    pixel_color = rgb_to_color([color_value, color_value, color_value])
                    brightness = color_value  # For text color determination

                square = Square(
                    side_length=self.pixel_size,
                    fill_color=pixel_color,
                    fill_opacity=1,
                    stroke_color=COLORS["pixel_grid"],
                    stroke_width=1,
                )
                square.move_to([j * self.pixel_size, -i * self.pixel_size, 0])
                grid.add(square)

                # Store reference to the square for later access
                self.pixel_map[(i, j)] = square

                # Add value text if needed
                if self.with_values:
                    if is_color:
                        # For color, just show a number
                        label = f"{i},{j}"
                    else:
                        # For grayscale, show the value
                        label = f"{int(data_tensor[i, j].item())}"

                    value_text = Text(
                        label, font_size=14, color=BLACK if brightness > 0.5 else WHITE
                    )
                    value_text.scale(0.2)
                    value_text.move_to(square.get_center())
                    grid.add(value_text)

        self.grid = grid
        return grid

    def highlight_pixel(
        self, position, color=COLORS["highlight"]["query"], opacity=0.7, z_index=10
    ):
        """Highlight a specific pixel in the grid"""
        row, col = position
        square = self.pixel_map.get((row, col))

        if square:
            # Create a highlight effect
            highlight = square.copy()
            highlight.set_fill(color, opacity=opacity)
            highlight.set_stroke(color, width=2)
            highlight.z_index = z_index

            return highlight
        return None

    def get_neighborhood(self, center, window_size):
        """Get coordinates within a neighborhood window"""
        row, col = center
        radius = window_size // 2

        # Get neighbor indices within radius
        neighbors = []
        for i in range(max(0, row - radius), min(self.grid_size, row + radius + 1)):
            for j in range(max(0, col - radius), min(self.grid_size, col + radius + 1)):
                neighbors.append((i, j))

        return neighbors

    def highlight_neighborhood(
        self,
        center,
        window_size,
        color=COLORS["highlight"]["neighborhood"],
        opacity=0.5,
    ):
        """Highlight the neighborhood of a pixel"""
        neighborhood = self.get_neighborhood(center, window_size)
        highlights = VGroup()

        for pos in neighborhood:
            # Skip center for neighborhood highlights
            if pos == center:
                continue

            highlight = self.highlight_pixel(pos, color, opacity)
            if highlight:
                highlights.add(highlight)

        return highlights

    def highlight_window(
        self, center, window_size, color=COLORS["highlight"]["window"]
    ):
        """Create an outline around the window area"""
        row, col = center
        radius = window_size // 2

        # Calculate window boundaries
        start_row = max(0, row - radius)
        start_col = max(0, col - radius)
        end_row = min(self.grid_size - 1, row + radius)
        end_col = min(self.grid_size - 1, col + radius)

        # Create rectangle around the window
        top_left_pixel = self.pixel_map.get((start_row, start_col))
        bottom_right_pixel = self.pixel_map.get((end_row, end_col))

        if top_left_pixel and bottom_right_pixel:
            top_left = top_left_pixel.get_corner(UL)
            bottom_right = bottom_right_pixel.get_corner(DR)

            window_rect = Rectangle(
                width=(end_col - start_col + 1) * self.pixel_size,
                height=(end_row - start_row + 1) * self.pixel_size,
                stroke_color=color,
                stroke_width=3,
                fill_opacity=0,
            )
            window_rect.move_to((top_left + bottom_right) / 2)

            return window_rect

        return None

    def get_row_major_path(self):
        """Get coordinates for a row-major traversal"""
        path = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                path.append((i, j))
        return path

    def get_morton_path(self):
        """Get coordinates for a Morton (Z-order) traversal"""
        coordinates = []
        # Generate all coordinates with their Morton codes
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                morton_code = morton_encode(j, i)  # Using j,i to match x,y convention
                coordinates.append((i, j, morton_code))

        # Sort by Morton code
        coordinates.sort(key=lambda x: x[2])

        # Return just the coordinates
        return [(i, j) for i, j, _ in coordinates]

    def create_path_line(self, path, color=COLORS["highlight"]["path"]):
        """Create a line connecting points on the path"""
        points = []
        for i, j in path:
            if (i, j) in self.pixel_map:
                square = self.pixel_map[(i, j)]
                points.append(square.get_center())

        line = VMobject()
        line.set_points_as_corners(points)
        line.set_stroke(color, width=2)

        return line


class NattenBasicVisualization(Slide):
    def setup(self):
        """Initialize grid and parameters"""
        self.grid_size = 100  # 16x16 grid for visualization
        self.window_size = 8  # 3x3 window
        # Initialize a single pixel helper for this visualization
        self.pixel_helper = PixelGrid(
            self.grid_size, pixel_size=0.05, with_values=False
        )

        # Load image from file
        img = Image.open("/Users/drissguessous/Code/vizz/media/images/image.png")
        img = img.resize((self.grid_size, self.grid_size))
        img_array = np.array(img)

        # Keep the color channels (don't convert to grayscale)
        self.image_tensor = torch.tensor(img_array)

        print(f"Loaded image with shape: {self.image_tensor.shape}")

    def construct(self):
        """Create the main visualization"""
        # Create title
        title = Text(
            "Neighborhood Attention (NATTEN)", font_size=36, color=COLORS["text"]
        ).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Create and display image grid
        image_grid = self.pixel_helper.create_grid(self.image_tensor)
        image_grid.scale(1.2).center()

        self.play(
            FadeIn(image_grid),
        )
        self.next_slide()

        # Add explanation of neighborhood attention
        explanation = Text(
            "NATTEN restricts attention to a local window around each query pixel",
            font_size=20,
            color=COLORS["text"],
            line_spacing=1.2,
        ).next_to(image_grid, DOWN, buff=0.5)

        self.play(Write(explanation))
        self.next_slide()

        # Demonstrate sliding window with a sequence of query positions
        demo_positions = [
            (25, 25),  # Top left area
            (50, 50),  # Top right area
            (75, 75),  # Bottom left area
        ]

        current_highlights = None
        current_window = None
        query_highlight = None

        query_label = Text(
            "Query Pixel", font_size=18, color=COLORS["highlight"]["query"]
        )
        query_label.next_to(image_grid, RIGHT, buff=0.5).shift(UP)

        neighborhood_label = Text(
            f"Local {self.window_size}x{self.window_size} Window",
            font_size=18,
            color=COLORS["highlight"]["neighborhood"],
        )
        neighborhood_label.next_to(query_label, DOWN, buff=0.5)

        self.play(Write(query_label), Write(neighborhood_label))

        for pos in demo_positions:
            # Remove previous highlights
            if current_highlights:
                self.play(FadeOut(current_highlights))
                current_highlights = None

            if current_window:
                self.play(FadeOut(current_window))
                current_window = None

            if query_highlight:
                self.play(FadeOut(query_highlight))
                query_highlight = None

            # Add new highlights for the current position
            query_highlight = self.pixel_helper.highlight_pixel(
                pos, COLORS["highlight"]["query"]
            )
            current_highlights = self.pixel_helper.highlight_neighborhood(
                pos, self.window_size
            )
            current_window = self.pixel_helper.highlight_window(pos, self.window_size)

            self.play(
                FadeIn(query_highlight),
                FadeIn(current_highlights),
                FadeIn(current_window),
            )
            self.next_slide()

        # Clean up for next section
        self.play(
            FadeOut(query_label),
            FadeOut(current_highlights),
            FadeOut(query_highlight),
            FadeOut(current_window),
            FadeOut(neighborhood_label),
        )


class RasterizationComparison(Slide):
    def setup(self):
        """Initialize grid and parameters"""
        self.grid_size = 6  # Grid size for visualization
        self.window_size = 3  # Window size for attention
        self.pixel_grid_helper = PixelGrid(
            self.grid_size, pixel_size=0.3, with_values=False
        )

        # Load image from file
        img = Image.open("/Users/drissguessous/Code/vizz/media/images/image.png")
        img = img.resize((self.grid_size, self.grid_size))
        img_array = np.array(img)

        # Keep the color channels (don't convert to grayscale)
        self.image_tensor = torch.tensor(img_array)
        print(f"Loaded image with shape: {self.image_tensor.shape}")

        # Define available rasterization orders
        self.rasterization_orders = {
            "row_major": {
                "label": "Row-Major Rasterization",
                "path_func": self.pixel_grid_helper.get_row_major_path,
            },
            "morton": {
                "label": "Morton Order (Z-curve)",
                "path_func": self.pixel_grid_helper.get_morton_path,
            },
            # Add more rasterization orders here as needed
        }

    def construct(self):
        """Create the main visualization"""
        # Choose rasterization order here
        rasterization_order = os.environ.get("ORDER", "row_major")

        # Create title
        title = Text(
            "Rasterization Patterns for NATTEN", font_size=36, color=COLORS["text"]
        ).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Create the grid based on the selected rasterization order
        grid_data = self.create_grid(rasterization_order)
        grid = grid_data["grid"]
        label = grid_data["label"]
        path = grid_data["path"]

        self.play(
            FadeIn(grid),
            Write(label),
        )
        self.next_slide()

        # Generate traversal paths
        self.play(FadeOut(title))
        col_squares, col_labels, row_squares = self.create_1d_layout_transformation(
            grid, path, LEFT, label=label
        )
        self.next_slide()

        # Visualize the NATTEN attention pattern
        attention_matrix, attention_title, attention_legend = (
            self.visualize_natten_attention(
                col_squares, row_squares, rasterization_order, col_labels
            )
        )
        self.next_slide()

        # Create text elements with different styles
        header_text = Text(
            "For image_size = 128x128\nwindow_size = 16x16\ntile_size = 128:",
            font_size=20,
            color=COLORS["text"],
        )

        if rasterization_order == "row_major":
            sparsity_value = Text(
                "86.72% blockwise sparsity",
                font_size=20,
                color=COLORS["text"],
                weight=BOLD,
            )
        elif rasterization_order == "morton":
            sparsity_value = Text(
                "93.55% blockwise sparsity",
                font_size=20,
                color=COLORS["text"],
                weight=BOLD,
            )

        # Group them together vertically
        sparsity_info = VGroup(header_text, sparsity_value)
        sparsity_info.arrange(DOWN, aligned_edge=LEFT)
        sparsity_info.scale(0.8)  # Scale down the entire group if needed

        # Position it under the legend
        sparsity_info.next_to(attention_legend, DOWN, buff=0.5)

        # Add it to the scene
        self.play(FadeIn(sparsity_info))

        self.next_slide()

    def create_grid(self, rasterization_order):
        """
        Create a grid with the specified rasterization order

        Parameters:
        -----------
        rasterization_order : str
            The rasterization order to use

        Returns:
        --------
        dict
            Dictionary containing the grid, label, and path
        """
        if rasterization_order not in self.rasterization_orders:
            raise ValueError(f"Unknown rasterization order: {rasterization_order}")

        order_info = self.rasterization_orders[rasterization_order]

        # Create the grid
        grid = self.pixel_grid_helper.create_grid(self.image_tensor)
        grid.scale(0.9).shift(UP)

        # Create the label
        label = Text(order_info["label"], font_size=22, color=COLORS["text"])
        label.next_to(grid, UP, buff=0.25)

        # Get the traversal path
        path = order_info["path_func"]()

        return {"grid": grid, "label": label, "path": path}

    def create_1d_layout_transformation(self, grid, path, side, label):
        """
        Transform a 2D grid to both 1D column and row layouts based on the provided traversal path

        Parameters:
        -----------
        grid : VGroup
            The 2D grid of pixels
        path : list
            List of (row, col) coordinates in the desired traversal order
        side : np.ndarray or manim.constants
            Position anchor (LEFT or RIGHT) for placement
        label : Text
            Label for the original grid

        Returns:
        --------
        tuple
            Tuple containing (column_squares, column_labels, row_squares)
        """
        # Get the original pixel squares from the grid
        original_squares = [square for square in grid if isinstance(square, Square)]

        # Create mapping from position to square
        position_to_square = {}
        for pos, pixel in self.pixel_grid_helper.pixel_map.items():
            if pixel in original_squares:
                position_to_square[pos] = pixel

        # Get pixel size
        pixel_size = self.pixel_grid_helper.pixel_size
        smaller_scale = 0.4  # Scale factor to make pixels smaller

        # Position for the 1D column, offset to the side of the original grid
        column_x = 3 * side[0]  # Further to the side
        column_top_y = 3  # Near the top of the screen

        # Create a VGroup to hold the transformed column squares
        transformed_squares = VGroup()

        # Add the squares in path order for the column
        for pos in path:
            if pos in position_to_square:
                # Get the square and add a copy to our transformed group
                square = position_to_square[pos]
                square_copy = square.copy().scale(smaller_scale)
                transformed_squares.add(square_copy)

        # Arrange the squares in a vertical column with proper spacing
        transformed_squares.arrange(
            DOWN,
            buff=pixel_size * 0.25,  # More consistent spacing based on pixel size
            center=False,
        )

        # Position the column at the desired location
        transformed_squares.move_to([column_x, column_top_y, 0], aligned_edge=UP)

        # Create transforms from original squares to arranged copies
        transforms = [FadeOut(label)]
        for i, pos in enumerate(path):
            if pos in position_to_square:
                square = position_to_square[pos]
                target = transformed_squares[i]
                transforms.append(square.animate.become(target))

        # Create index labels relative to the arranged squares
        index_labels = VGroup()
        for i, square in enumerate(transformed_squares):
            label = Text(f"{i}", font_size=12, color=COLORS["text"])
            label.next_to(square, LEFT, buff=pixel_size * 0.5)
            index_labels.add(label)

        # Play the transformation animation
        self.play(*transforms, run_time=2, rate_func=smooth)

        # Add the labels after the transformation is complete
        self.play(FadeIn(index_labels))

        # Now create the row layout directly above the column
        row_squares = VGroup()

        # Create copies of the column squares for the row
        for square in transformed_squares:
            square_copy = square.copy()
            row_squares.add(square_copy)

        # Arrange the squares in a horizontal row with proper spacing
        row_squares.arrange(RIGHT, buff=pixel_size * 0.25, center=False)

        # Calculate row position precisely above the first column element
        first_square = transformed_squares[0]

        # Position the row so its leftmost element is directly above the first column element
        row_squares.move_to(
            first_square.get_left() + np.array([0, 1 * pixel_size, 0]),
            aligned_edge=DOWN + LEFT,
        )

        self.play(FadeIn(row_squares), run_time=1)

        return transformed_squares, index_labels, row_squares

    def visualize_natten_attention(
        self, col_squares, row_squares, rasterization_order, col_labels
    ):
        """
        Visualize the NATTEN attention mask pattern using the row and column vectors

        Parameters:
        -----------
        col_squares : VGroup
            The column vector squares
        row_squares : VGroup
            The row vector squares
        rasterization_order : str
            The rasterization order being used

        Returns:
        --------
        tuple
            Tuple containing (attention_matrix, title, legend)
        """
        from attn_gym.masks.natten import generate_natten, generate_morton_natten

        if rasterization_order == "row_major":
            mask_func = generate_natten(
                self.grid_size, self.grid_size, self.window_size, self.window_size
            )
        else:
            mask_func = generate_morton_natten(
                self.grid_size, self.grid_size, self.window_size, self.window_size
            )
        # Create a matrix to hold attention connections
        attention_matrix = VGroup()

        # Calculate pixel size based on existing squares
        pixel_size = col_squares[0].width

        # Create the background matrix of attention scores
        for i in range(len(col_squares)):
            row = VGroup()
            for j in range(len(row_squares)):
                # Create a square for each attention score position
                square = Square(
                    side_length=pixel_size,
                    fill_opacity=0.8,
                    stroke_width=1,
                    stroke_color=COLORS["text"],
                )

                # Position the square in the grid
                square.move_to(
                    col_squares[i].get_center()
                    + np.array(
                        [
                            row_squares[j].get_center()[0]
                            - row_squares[0].get_center()[0],
                            0,
                            0,
                        ]
                    )
                )
                row.add(square)
            attention_matrix.add(row)

        # Color the attention matrix based on the NATTEN pattern
        for i in range(len(col_squares)):
            for j in range(len(row_squares)):
                if mask_func(0, 0, torch.tensor(i), torch.tensor(j)):
                    # Yellow for active connections (within kernel)
                    attention_matrix[i][j].set_fill(YELLOW)
                else:
                    # Blue for masked out connections (outside kernel)
                    attention_matrix[i][j].set_fill(BLUE)

        # Create labels for better understanding
        order_display_name = self.rasterization_orders[rasterization_order]["label"]
        title = Text(
            f"NATTEN Attention Pattern ({order_display_name})",
            font_size=24,
            color=COLORS["text"],
        )
        title.to_edge(UP)

        legend = VGroup()

        # Yellow box for active tokens
        active_box = Square(
            side_length=0.3,
            fill_color=YELLOW,
            fill_opacity=0.8,
            stroke_color=COLORS["text"],
        )
        active_label = Text("Active tokens", font_size=16, color=COLORS["text"])
        active_label.next_to(active_box, RIGHT, buff=0.2)
        active_group = VGroup(active_box, active_label)

        # Blue box for masked tokens
        masked_box = Square(
            side_length=0.3,
            fill_color=BLUE,
            fill_opacity=0.8,
            stroke_color=COLORS["text"],
        )
        masked_label = Text("Masked tokens", font_size=16, color=COLORS["text"])
        masked_label.next_to(masked_box, RIGHT, buff=0.2)
        masked_group = VGroup(masked_box, masked_label)

        # Arrange legend items
        legend.add(active_group, masked_group)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        legend.to_edge(RIGHT)

        # Animate the appearance of the attention matrix
        self.play(
            FadeIn(attention_matrix),
            FadeOut(col_squares),
            FadeOut(row_squares),
            FadeOut(col_labels),
        )
        self.play(FadeIn(title), FadeIn(legend), run_time=1.0)

        return attention_matrix, title, legend

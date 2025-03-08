"""
To run call: manimgl natten_visualization.py RasterizationComparison
"""

import torch
import numpy as np
from manim import *
from manim_slides import Slide
from PIL import Image
from attn_gym.masks.natten import morton_encode

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
        self.grid_size = 6  # 16x16 grid for visualization
        self.window_size = 3  # 3x3 window
        self.standard_helper = PixelGrid(
            self.grid_size, pixel_size=0.3, with_values=False
        )
        self.morton_helper = PixelGrid(
            self.grid_size, pixel_size=0.3, with_values=False
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
            "Rasterization Patterns for NATTEN", font_size=36, color=COLORS["text"]
        ).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Create and display image grids side by side
        standard_grid = self.standard_helper.create_grid(self.image_tensor)
        morton_grid = self.morton_helper.create_grid(self.image_tensor)

        standard_grid.scale(0.9).to_edge(LEFT, buff=1)
        morton_grid.scale(0.9).to_edge(RIGHT, buff=1)

        standard_label = Text(
            "Row-Major Rasterization", font_size=22, color=COLORS["text"]
        )
        morton_label = Text(
            "Morton Order (Z-curve)", font_size=22, color=COLORS["text"]
        )

        standard_label.next_to(standard_grid, UP, buff=0.25)
        morton_label.next_to(morton_grid, UP, buff=0.25)

        self.play(
            FadeIn(standard_grid),
            FadeIn(morton_grid),
            Write(standard_label),
            Write(morton_label),
        )
        self.next_slide()

        # Generate traversal paths
        row_major_path = self.standard_helper.get_row_major_path()
        morton_path = self.morton_helper.get_morton_path()

        self.create_1d_layout_transformation(standard_grid, row_major_path, LEFT)

        self.create_1d_layout_transformation(morton_grid, morton_path, RIGHT)
        self.next_slide()

    def create_1d_layout_transformation(self, grid, path, side):
        """
        Transform a 2D grid to a 1D column layout based on the provided traversal path

        Parameters:
        -----------
        grid : VGroup
            The 2D grid of pixels
        path : list
            List of (row, col) coordinates in the desired traversal order
        label_text : str
            Label for the 1D layout
        side : np.ndarray or manim.constants
            Position anchor (LEFT or RIGHT) for placement
        """
        # Create a target 1D layout as a column
        target_positions = {}

        # Get the original pixel squares from the grid
        original_squares = [square for square in grid if isinstance(square, Square)]

        # Create mapping from position to square
        position_to_square = {}
        for i, square in enumerate(original_squares):
            # We know each pixel grid has a mapping from (row, col) to square
            # So we'll reverse-map by iterating through the pixel_map
            pixel_map = (
                self.standard_helper.pixel_map
                if side is LEFT
                else self.morton_helper.pixel_map
            )
            for pos, pixel in pixel_map.items():
                if pixel == square:
                    position_to_square[pos] = square
                    break

        # Calculate spacing based on pixel size
        pixel_size = (
            self.standard_helper.pixel_size
            if side is LEFT
            else self.morton_helper.pixel_size
        )
        spacing = pixel_size * 0.15  # Reduced spacing for smaller pixels
        smaller_scale = 0.4  # Scale factor to make pixels smaller

        # Position for the 1D column, offset to the side of the original grid
        column_x = 3 * side[0]  # Further to the side
        column_top_y = 0  # Near the top of the screen

        # Create target positions for each square in the traversal order
        for i, pos in enumerate(path):
            if pos in position_to_square:
                square = position_to_square[pos]
                # Calculate target position in the column
                target_pos = np.array([column_x, column_top_y - i * spacing, 0])
                target_positions[square] = target_pos

        # Create index labels for the flattened array
        index_labels = VGroup()
        for i, pos in enumerate(path):
            if pos in position_to_square:
                label = Text(f"{i}", font_size=12, color=COLORS["text"])
                label.move_to(
                    [column_x - pixel_size * 1.5, column_top_y - i * spacing, 0]
                )
                index_labels.add(label)

        # Now create the transformations for all squares
        transforms = []
        for square in original_squares:
            if square in target_positions:
                # Scale down the square and move it to the target position
                transforms.append(
                    square.animate.scale(smaller_scale).move_to(
                        target_positions[square]
                    )
                )

        # Play the transformation animation
        self.play(*transforms, run_time=2.5, rate_func=smooth)

        # # Show the indices
        # self.play(Write(index_labels))

        # # Highlight the sequential accesses
        # highlights = VGroup()
        # for i in range(len(path) - 1):
        #     pos = path[i]
        #     next_pos = path[i + 1]

        #     if pos in position_to_square and next_pos in position_to_square:
        #         current_square = position_to_square[pos]
        #         next_square = position_to_square[next_pos]

        #         arrow = Arrow(
        #             current_square.get_center(),
        #             next_square.get_center(),
        #             buff=0.1,
        #             color=COLORS["highlight"]["path"],
        #             max_tip_length_to_length_ratio=0.15,
        #             stroke_width=2
        #         )
        #         highlights.add(arrow)

        # # Display the first few arrows to show the access pattern
        # first_few = highlights[:min(5, len(highlights))]
        # self.play(ShowCreation(first_few), run_time=1.5)
        # self.wait(0.5)
        # self.play(FadeOut(first_few))

        return index_labels

"""
To run call: manimgl ordering_comparison.py OrderingPatterns
"""

import torch
from manim import *
from manim_slides import Slide
from vizz.flex.natten import COLORS, PixelGrid, morton_encode

# Custom configuration to use a light theme for presentations
config.background_color = WHITE


class OrderingPatterns(Slide):
    def setup(self):
        """Initialize grid and parameters"""
        self.grid_size = 8  # An 8x8 grid for clearer numbers
        self.pixel_size = 0.5
        self.row_major_helper = PixelGrid(
            self.grid_size, pixel_size=self.pixel_size, with_values=False
        )
        self.column_major_helper = PixelGrid(
            self.grid_size, pixel_size=self.pixel_size, with_values=False
        )
        self.morton_helper = PixelGrid(
            self.grid_size, pixel_size=self.pixel_size, with_values=False
        )
        self.hilbert_helper = PixelGrid(
            self.grid_size, pixel_size=self.pixel_size, with_values=False
        )

        # Create a simple pattern for visualization
        self.pattern_tensor = torch.zeros((self.grid_size, self.grid_size))

    def construct(self):
        """Create the main visualization"""
        # Create title
        title = Text(
            "1D Ordering Patterns for 2D Data", font_size=36, color=COLORS["text"]
        ).to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Create grids
        row_grid = self.row_major_helper.create_grid(self.pattern_tensor)
        column_grid = self.column_major_helper.create_grid(self.pattern_tensor)
        morton_grid = self.morton_helper.create_grid(self.pattern_tensor)
        hilbert_grid = self.hilbert_helper.create_grid(self.pattern_tensor)

        # Position grids in a 2x2 layout
        row_grid.scale(0.8).to_corner(UL, buff=1.5)
        column_grid.scale(0.8).to_corner(UR, buff=1.5)
        morton_grid.scale(0.8).to_corner(DL, buff=1.5)
        hilbert_grid.scale(0.8).to_corner(DR, buff=1.5)

        # Add labels
        row_label = Text("Row-Major Order", font_size=20, color=COLORS["text"])
        column_label = Text("Column-Major Order", font_size=20, color=COLORS["text"])
        morton_label = Text("Morton Z-Order", font_size=20, color=COLORS["text"])
        hilbert_label = Text("Hilbert Curve", font_size=20, color=COLORS["text"])

        row_label.next_to(row_grid, UP, buff=0.3)
        column_label.next_to(column_grid, UP, buff=0.3)
        morton_label.next_to(morton_grid, UP, buff=0.3)
        hilbert_label.next_to(hilbert_grid, UP, buff=0.3)

        self.play(
            FadeIn(row_grid),
            FadeIn(column_grid),
            FadeIn(morton_grid),
            FadeIn(hilbert_grid),
            Write(row_label),
            Write(column_label),
            Write(morton_label),
            Write(hilbert_label),
        )
        self.next_slide()

        # Generate different orders
        row_major_path = self.get_row_major_path()
        column_major_path = self.get_column_major_path()
        morton_path = self.get_morton_path()
        hilbert_path = self.get_hilbert_path()

        # Create 1D order visualizations
        row_order_vis = self.row_major_helper.create_order_visualization(row_major_path)
        column_order_vis = self.column_major_helper.create_order_visualization(
            column_major_path
        )
        morton_order_vis = self.morton_helper.create_order_visualization(morton_path)
        hilbert_order_vis = self.hilbert_helper.create_order_visualization(hilbert_path)

        self.play(
            FadeIn(row_order_vis),
            FadeIn(column_order_vis),
            FadeIn(morton_order_vis),
            FadeIn(hilbert_order_vis),
        )
        self.next_slide()

        # Add explanation text
        explanation = VGroup(
            Text("Memory Locality Properties:", font_size=22, color=COLORS["text"]),
            Text(
                "• Row/Column-Major: Good for 1D operations, poor at 2D boundaries",
                font_size=18,
                color=COLORS["text"],
            ),
            Text(
                "• Z-Order: Better 2D locality, efficient for hierarchical operations",
                font_size=18,
                color=COLORS["text"],
            ),
            Text(
                "• Hilbert: Optimal locality, each step moves to adjacent cell",
                font_size=18,
                color=COLORS["text"],
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        explanation.move_to(ORIGIN)

        self.play(
            FadeOut(row_order_vis),
            FadeOut(column_order_vis),
            FadeOut(morton_order_vis),
            FadeOut(hilbert_order_vis),
            Write(explanation),
        )
        self.next_slide()

        # Clean up for conclusion
        self.play(FadeOut(explanation))

        conclusion = Text(
            "Spatial locality in memory access patterns significantly impacts\n"
            "performance for operations like sliding window attention",
            font_size=20,
            color=COLORS["text"],
            line_spacing=1.2,
        ).move_to(ORIGIN)

        self.play(Write(conclusion))
        self.next_slide()

    def get_row_major_path(self):
        """Get coordinates for a row-major traversal"""
        path = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                path.append((i, j))
        return path

    def get_column_major_path(self):
        """Get coordinates for a column-major traversal"""
        path = []
        for j in range(self.grid_size):
            for i in range(self.grid_size):
                path.append((i, j))
        return path

    def get_morton_path(self):
        """Get coordinates for a Morton (Z-order) traversal"""
        coordinates = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                morton_code = morton_encode(j, i)
                coordinates.append((i, j, morton_code))

        coordinates.sort(key=lambda x: x[2])
        return [(i, j) for i, j, _ in coordinates]

    # This is a simplified Hilbert curve implementation for powers of 2
    def get_hilbert_path(self):
        """Calculate Hilbert curve coordinates for a grid"""
        # Only works correctly for powers of 2
        if self.grid_size & (self.grid_size - 1) != 0:
            # For non-powers of 2, return a simpler pattern
            return self.get_morton_path()

        n = self.grid_size
        path = []

        def d2xy(n, d):
            """Convert d to (x,y) coordinates of the Hilbert curve"""
            assert d <= n * n - 1
            t = d
            x = y = 0
            s = 1
            while s < n:
                rx = 1 & (t // 2)
                ry = 1 & (t ^ rx)
                x, y = rot(s, x, y, rx, ry)
                x += s * rx
                y += s * ry
                t //= 4
                s *= 2
            return x, y

        def rot(n, x, y, rx, ry):
            """Rotate/flip a quadrant appropriately"""
            if ry == 0:
                if rx == 1:
                    x = n - 1 - x
                    y = n - 1 - y
                # Swap x and y
                x, y = y, x
            return x, y

        for i in range(n * n):
            x, y = d2xy(n, i)
            path.append((y, x))  # Adjust to match our row, col convention

        return path


if __name__ == "__main__":
    # Use the OrderingPatterns class as the default scene
    pass

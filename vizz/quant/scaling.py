import torch
from manimlib import *


class Scaling(Scene):
    def per_tensor(self, matrix: torch.Tensor):
        pass


    def scan_rectangle(self, matrix, H, W):
        """Animates a rectangle of H,W shape over a matrix
        Calculating the abs_max value
        """

        # Get the position of the first element (top-left entry)
        first_element_pos = matrix.get_entries()[0].get_center()

        # Move the rectangle to that position
        rect = Rectangle(height=H, width=W, color=RED)
        rect.move_to(first_element_pos)
        self.play(ShowCreation(rect))

        # self.play(fade_out(rect))

        # for i in range(matrix.shape[0] - H + 1):
        #     for j in range(matrix.shape[1] - W + 1):
        #         sub_matrix = matrix[i:i+H, j:j+W]
        #         abs_max = torch.max(torch.abs(sub_matrix))
        #         self.play(rect.move_to, matrix.get_center() + np.array([j - matrix.shape[1] // 2 + W // 2, matrix.shape[0] // 2 - i - H // 2, 0]))
        #         self.wait(0.5)

    def construct(self):
        matrix = torch.arange(16).view(4,4)
        matrix = matrix.float()

        matrix = Matrix(matrix.tolist())
        self.play(ShowCreation(matrix))
        self.embed()
        # # Get the position of the first element (top-left entry)
        # first_element_pos = matrix.get_entries()[0].get_center()

        # # Move the rectangle to that position
        # H,W = 2,2
        # rect = Rectangle(height=H, width=W, color=RED)
        # rect.move_to(first_element_pos)
        # self.play(ShowCreation(rect))
        # self.play(FadeOut(rect))


        H, W = 3, 3  # Example dimensions

        # Get the positions of the corner elements
        def get_center(matrix, H, W):
            first_row = matrix.get_entries()[0]
            first_col = matrix.get_entries()[:][0]
            horizontal = first_row[0].get_center()[0] - first_row[H].get_center()[0]
            vertical = first_col[0].get_center()[1] - first_col[W].get_center()[1]
            return horizontal, vertical, 0

        # Get the element positions and calculate their spacing
        top_element = matrix.get_entries()[0][0]
        right_element = matrix.get_entries()[0][1]
        bottom_element = matrix.get_entries()[1][0]

        # Calculate spacings - including the element size itself, not just gap
        h_units = right_element.get_width()
        v_units = bottom_element.get_height()

        print(v_units)
        print(h_units)
        rect = Rectangle(
            height=H * v_units,  # H units tall
            width=W * H,   # W units wide
            color=RED
        )
        new_center = get_center(matrix, H, W)
        print(new_center)
        rect.move_to(new_center)
        self.play(ShowCreation(rect))
        self.play(FadeOut(rect))

"""Matrix formatting utilities for cleaner visualizations."""


def insert_ellipsis_columns(matrix_data, block_size=32, ellipsis_symbol="\\cdots"):
    """Insert ellipsis columns between blocks for cleaner matrix display.

    Args:
        matrix_data: 2D numpy array of numeric values
        block_size: Size of each block (default 32)
        ellipsis_symbol: LaTeX symbol to use for ellipsis (default \\cdots)

    Returns:
        2D list with ellipsis columns inserted between blocks.
        Can be used with Manim's Matrix class using MathTex for rendering.

    Example:
        Input:  [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]  (12 columns, block_size=4)
        Output: [[1, 2, 3, 4, '\\cdots', 5, 6, 7, 8, '\\cdots', 9, 10, 11, 12]]
    """
    num_rows, num_cols = matrix_data.shape
    num_blocks = num_cols // block_size

    result = []
    for row_idx in range(num_rows):
        row_with_ellipsis = []

        for block_idx in range(num_blocks):
            # Add block values
            start_idx = block_idx * block_size
            end_idx = start_idx + block_size

            for col_idx in range(start_idx, end_idx):
                # Format number to 1 decimal place
                value = matrix_data[row_idx, col_idx]
                row_with_ellipsis.append(f"{value:.1f}")

            # Add ellipsis after each block except the last
            if block_idx < num_blocks - 1:
                row_with_ellipsis.append(ellipsis_symbol)

        result.append(row_with_ellipsis)

    return result


def map_ellipsis_indices(original_cols, block_size=32):
    """Create a mapping from display indices (with ellipsis) to original data indices.

    Args:
        original_cols: Number of columns in the original matrix
        block_size: Size of each block (default 32)

    Returns:
        dict mapping display_index -> original_index (or None for ellipsis columns)

    Example:
        For 96 columns with block_size=32:
        {0: 0, 1: 1, ..., 31: 31, 32: None, 33: 32, ..., 64: 63, 65: None, 66: 64, ...}
    """
    num_blocks = original_cols // block_size
    mapping = {}
    display_idx = 0

    for block_idx in range(num_blocks):
        # Map block values
        start_idx = block_idx * block_size
        end_idx = start_idx + block_size

        for original_idx in range(start_idx, end_idx):
            mapping[display_idx] = original_idx
            display_idx += 1

        # Ellipsis column (except after last block)
        if block_idx < num_blocks - 1:
            mapping[display_idx] = None  # Ellipsis has no original index
            display_idx += 1

    return mapping


def get_display_indices_for_block(block_idx, block_size=32):
    """Get the display matrix indices for a given block index.

    Args:
        block_idx: Which block (0-indexed)
        block_size: Size of each block (default 32)

    Returns:
        tuple of (start_display_idx, end_display_idx) for use with matrix.get_entries()

    Example:
        For block_size=32:
        Block 0: (0, 32)    - display indices 0-31
        Block 1: (33, 65)   - display indices 33-64 (skipping ellipsis at 32)
        Block 2: (66, 98)   - display indices 66-97 (skipping ellipsis at 65)
    """
    # Each block takes block_size display positions
    # Plus 1 for each ellipsis before this block
    start = block_idx * (block_size + 1)
    end = start + block_size
    return (start, end)

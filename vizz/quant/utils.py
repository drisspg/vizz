"""Utility functions for floating point quantization visualizations.

This module provides reusable quantization primitives used across different
visualization scenes, keeping the scene code focused on the animations rather
than the quantization math.
"""

import torch
from typing import Tuple, Literal


# ============ QUANTIZATION CONSTANTS ============

# FP8 E4M3 constants
FP8_E4M3_MAX = 448.0
FP8_E4M3_MAX_POW2 = 8  # 256

# FP4 E2M1 constants
FP4_E2M1_MAX = 6.0
FP4_E2M1_MAX_POW2 = 2  # 4

# E8M0 (power-of-2) constants
E8M0_BIAS = 127
E8M0_MIN_EXPONENT = -126

# E4M3 (for scale quantization) constants
E4M3_EPS = 1.5259e-05
E4M3_MAX = 448.0


# ============ TARGET DTYPE CONFIGS ============


class TargetDtypeConfig:
    """Configuration for target quantization dtype."""

    def __init__(
        self,
        max_value: float,
        max_pow2: int,
        torch_dtype: torch.dtype,
        name: str,
        bits: int,
    ):
        self.max_value = max_value
        self.max_pow2 = max_pow2
        self.torch_dtype = torch_dtype
        self.name = name
        self.bits = bits


TARGET_DTYPE_CONFIGS = {
    "fp8": TargetDtypeConfig(
        max_value=FP8_E4M3_MAX,
        max_pow2=FP8_E4M3_MAX_POW2,
        torch_dtype=torch.float8_e4m3fn,
        name="FP8 E4M3",
        bits=8,
    ),
    "fp4": TargetDtypeConfig(
        max_value=FP4_E2M1_MAX,
        max_pow2=FP4_E2M1_MAX_POW2,
        torch_dtype=torch.uint8,  # Will pack later
        name="FP4 E2M1",
        bits=4,
    ),
}


SCALE_DTYPE_CONFIGS = {
    "e8m0": {"name": "E8M0 (power-of-2)", "bits": 8},
    "e4m3": {"name": "E4M3", "bits": 8},
    "fp32": {"name": "FP32", "bits": 32},
}


# ============ AMAX CALCULATION ============

ScaleType = Literal["per_tensor", "per_row", "per_block"]


def calculate_amax(
    tensor: torch.Tensor,
    scale_type: ScaleType,
    block_size: int = 2,
) -> Tuple[torch.Tensor, int]:
    """Calculate amax (maximum absolute value) based on granularity.

    Args:
        tensor: Input tensor [rows, cols]
        scale_type: Granularity of scales
        block_size: Size of blocks (only used for per_block)

    Returns:
        Tuple of (amax tensor, number of scales)
    """
    if scale_type == "per_tensor":
        amax = torch.max(torch.abs(tensor)).unsqueeze(0).unsqueeze(0)
        num_scales = 1
    elif scale_type == "per_row":
        amax = torch.max(torch.abs(tensor), dim=1).values.unsqueeze(1)
        num_scales = tensor.shape[0]
    elif scale_type == "per_block":
        rows, cols = tensor.shape
        if cols % block_size != 0:
            raise ValueError(
                f"cols ({cols}) must be divisible by block_size ({block_size})"
            )

        # Reshape to blocks: [rows, num_blocks, block_size]
        reshaped = tensor.reshape(rows, cols // block_size, block_size)
        amax = torch.max(torch.abs(reshaped), dim=-1).values
        num_scales = rows * (cols // block_size)
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    return amax, num_scales


# ============ SCALE CALCULATION ============


def calculate_e8m0_scale_rtne(
    amax_value: float,
    target_max_pow2: int,
) -> Tuple[float, int, int]:
    """Compute E8M0 scale from amax using RTNE (Round to Nearest Even) method.

    This replicates the logic from torchao's to_mx() function with ScaleCalculationMode.EVEN.
    RTNE = Round to Nearest, ties to Even.

    Args:
        amax_value: Maximum absolute value (scalar)
        target_max_pow2: Power-of-2 max value for target dtype (e.g., 8 for FP8 E4M3)

    Returns:
        Tuple of (fp32_scale, e8m0_biased, e8m0_unbiased):
            - fp32_scale: The FP32 scale factor (power of 2)
            - e8m0_biased: The E8M0 biased exponent byte (0-255)
            - e8m0_unbiased: The unbiased exponent (-127 to 127)
    """
    # Convert amax to float32 tensor
    max_abs = torch.tensor(amax_value, dtype=torch.float32)

    # Constants
    MBITS_F32 = 23
    MBITS_F8_E4M3 = 3
    EBITS_F32 = 8
    SBITS = 1
    F32_EXP_BIAS = 127

    # Apply EVEN rounding before extracting exponent
    # This rounds the mantissa to the target precision before determining the scale
    nan_mask = torch.isnan(max_abs)
    max_abs_int32 = max_abs.view(torch.int32)

    # Round to nearest even by adding to the mantissa
    val_to_add = 1 << (MBITS_F32 - MBITS_F8_E4M3 - 1)  # 1 << 19
    # Mask keeps sign + exponent, zeros out mantissa
    mask = ((1 << (EBITS_F32 + SBITS)) - 1) << MBITS_F32  # 0x1FF << 23
    max_abs_int32 = (max_abs_int32 + val_to_add) & mask
    max_abs = max_abs_int32.view(torch.float32)

    # Restore NaN if input was NaN
    if nan_mask.item():
        max_abs = torch.tensor(float("nan"), dtype=torch.float32)

    # Extract the power-of-2 exponent from the rounded float32 representation
    max_abs_int32 = max_abs.view(torch.int32)
    extracted_pow2 = (
        (torch.bitwise_right_shift(max_abs_int32, MBITS_F32)) & 0b11111111
    ) - F32_EXP_BIAS

    # Calculate the E8M0 scale exponent (unbiased)
    # For FP8 E4M3: scale_exponent = extracted_pow2 - target_max_pow2
    scale_e8m0_unbiased = extracted_pow2 - target_max_pow2

    # Clamp to the valid E8M0 range [-127, 127]
    scale_e8m0_unbiased = torch.clamp(
        scale_e8m0_unbiased, min=-E8M0_BIAS, max=E8M0_BIAS + 1
    )

    # Add bias to get the E8M0 biased representation
    scale_e8m0_biased = scale_e8m0_unbiased + E8M0_BIAS
    scale_e8m0_biased = scale_e8m0_biased.to(torch.uint8)

    # Handle NaN values (set to 255)
    if torch.isnan(torch.tensor(amax_value, dtype=torch.float32)):
        scale_e8m0_biased = torch.tensor(255, dtype=torch.uint8)

    # Extract scalar values
    scale_e8m0_biased_val = scale_e8m0_biased.item()
    scale_e8m0_unbiased_val = scale_e8m0_biased_val - E8M0_BIAS

    # Create FP32 scale from E8M0 exponent
    scale_fp32 = 2.0**scale_e8m0_unbiased_val

    return scale_fp32, scale_e8m0_biased_val, scale_e8m0_unbiased_val


def calculate_e8m0_scale(amax: torch.Tensor, target_max_pow2: int) -> torch.Tensor:
    """Calculate power-of-2 E8M0 scales (MX format).

    E8M0 format stores only the exponent (no mantissa), giving power-of-2 scales.
    This is used in MicroScaling (MX) formats.

    Args:
        amax: Maximum absolute values
        target_max_pow2: Power-of-2 max value for target dtype (e.g., 8 for FP8)

    Returns:
        FP32 scales that are exact powers of 2
    """
    amax_f32 = amax.to(torch.float32)

    # Extract exponent from float32 representation
    amax_int32 = amax_f32.view(torch.int32)
    extracted_pow2 = ((amax_int32 >> 23) & 0xFF) - 127

    # Calculate E8M0 unbiased exponent: scale = 2^(amax_exp - target_max_pow2)
    scale_e8m0_unbiased = extracted_pow2 - target_max_pow2
    scale_e8m0_unbiased = torch.clamp(
        scale_e8m0_unbiased, min=-E8M0_BIAS, max=E8M0_BIAS
    )

    # Create FP32 scale from E8M0 exponent
    scale_e8m0_biased = scale_e8m0_unbiased + E8M0_BIAS
    scale_fp32 = (scale_e8m0_biased.to(torch.int32) << 23).view(torch.float32)
    scale_fp32 = torch.clamp(scale_fp32, min=2**E8M0_MIN_EXPONENT)

    return scale_fp32


def calculate_e4m3_scale(amax: torch.Tensor, target_max: float) -> torch.Tensor:
    """Calculate E4M3 quantized scales (NVFP4 format).

    E4M3 format provides more precision than E8M0 by including a mantissa,
    but scales themselves are quantized to E4M3 precision.
    This is used in NVIDIA's FP4 format.

    Args:
        amax: Maximum absolute values
        target_max: Maximum value for target dtype (e.g., 448.0 for FP8)

    Returns:
        FP32 scales quantized through E4M3
    """
    # Calculate ideal scale
    scale_fp32 = amax.to(torch.float32) / target_max

    # Quantize to E4M3 and back to FP32
    scale_e4m3 = torch.clamp(scale_fp32, min=E4M3_EPS, max=E4M3_MAX).to(
        torch.float8_e4m3fn
    )

    scale_fp32 = scale_e4m3.to(torch.float32)

    return scale_fp32


def calculate_fp32_scale(amax: torch.Tensor, target_max: float) -> torch.Tensor:
    """Calculate simple FP32 scales.

    This is the baseline approach: scale = amax / target_max
    No quantization of the scale values themselves.

    Args:
        amax: Maximum absolute values
        target_max: Maximum value for target dtype

    Returns:
        FP32 scales
    """
    return amax.to(torch.float32) / target_max


def calculate_scale(
    amax: torch.Tensor,
    scale_dtype: Literal["e8m0", "e4m3", "fp32"],
    target_max: float,
    target_max_pow2: int,
) -> torch.Tensor:
    """Calculate scale factors for quantization.

    Unified interface for different scale calculation methods.

    Args:
        amax: Maximum absolute values
        scale_dtype: Method for calculating scales
        target_max: Maximum representable value in target dtype
        target_max_pow2: Power-of-2 approximation of target_max

    Returns:
        Scale factors in FP32
    """
    if scale_dtype == "e8m0":
        return calculate_e8m0_scale(amax, target_max_pow2)
    elif scale_dtype == "e4m3":
        return calculate_e4m3_scale(amax, target_max)
    elif scale_dtype == "fp32":
        return calculate_fp32_scale(amax, target_max)
    else:
        raise ValueError(f"Unknown scale_dtype: {scale_dtype}")


# ============ DATA QUANTIZATION ============


def broadcast_scales(
    scales: torch.Tensor,
    scale_type: ScaleType,
    target_shape: Tuple[int, int],
    block_size: int = 2,
) -> torch.Tensor:
    """Broadcast scale tensor to match data tensor shape.

    Args:
        scales: Scale values
        scale_type: Granularity of scales
        target_shape: Target shape [rows, cols]
        block_size: Size of blocks (only used for per_block)

    Returns:
        Broadcasted scales matching target_shape
    """
    if scale_type == "per_tensor":
        # Already broadcasts naturally
        return scales
    elif scale_type == "per_row":
        # Already has right shape [rows, 1]
        return scales
    elif scale_type == "per_block":
        # Repeat each scale block_size times along columns
        return scales.repeat_interleave(block_size, dim=1)
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")


def quantize_data(
    data: torch.Tensor,
    scales: torch.Tensor,
    scale_type: ScaleType,
    target_max: float,
    block_size: int = 2,
) -> torch.Tensor:
    """Quantize data using calculated scales.

    Performs: quantized = clamp(data / scale, -target_max, target_max)

    Args:
        data: Input tensor [rows, cols]
        scales: Scale factors (shape depends on scale_type)
        scale_type: Granularity of scales
        target_max: Maximum value in target dtype
        block_size: Size of blocks (only used for per_block)

    Returns:
        Quantized data in FP32 (not cast to target dtype yet)
    """
    data_f32 = data.to(torch.float32)

    # Broadcast scales to match data shape
    scale_broadcast = broadcast_scales(scales, scale_type, data.shape, block_size)

    # Quantize: divide by scale and clamp
    data_scaled = data_f32 / scale_broadcast
    data_quantized = torch.clamp(data_scaled, -target_max, target_max)

    return data_quantized


# ============ COMPRESSION METRICS ============


def calculate_compression_ratio(
    num_elements: int,
    num_scales: int,
    target_bits: int,
    scale_bits: int,
    original_bits: int = 16,
) -> float:
    """Calculate compression ratio for quantization scheme.

    Args:
        num_elements: Total number of data elements
        num_scales: Number of scale factors
        target_bits: Bits per quantized element
        scale_bits: Bits per scale factor
        original_bits: Bits per original element (default BF16)

    Returns:
        Compression ratio (original_size / compressed_size)
    """
    original_size = num_elements * original_bits
    compressed_size = num_elements * target_bits + num_scales * scale_bits
    return original_size / compressed_size

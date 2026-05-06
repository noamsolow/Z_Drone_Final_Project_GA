"""
Relative depth estimation adapters.

The real implementation uses the Depth -> Mask strategy:

1. Run monocular depth estimation on the full original image.
2. Return a full-resolution relative depth map.
3. Let downstream code extract/mask the drone region from that depth map.

Running depth on the full image preserves scene context, which is important for
monocular depth models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    torch = None


DEFAULT_DEPTH_ANYTHING_V2_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def _to_pil_rgb_image(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """Normalize supported image inputs into a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened_image:
            return opened_image.convert("RGB").copy()

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim not in (2, 3):
            raise ValueError("Expected a 2D or 3D image array, got shape {}".format(array.shape))

        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            if array.size > 0 and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(array).convert("RGB")

    raise TypeError("Unsupported image type: {}".format(type(image).__name__))


def _target_size_from_image(image: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[int, int]:
    """Return ``(height, width)`` for supported image inputs."""
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened_image:
            width, height = opened_image.size
        return height, width

    if isinstance(image, np.ndarray):
        if image.ndim < 2:
            raise ValueError("Expected image array with at least 2 dimensions.")
        return int(image.shape[0]), int(image.shape[1])

    pil_image = _to_pil_rgb_image(image)
    width, height = pil_image.size
    return height, width


def _tensor_dict_to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Move tensor values from a processor output dictionary to a device."""
    moved_inputs = {}
    for key, value in inputs.items():
        moved_inputs[key] = value.to(device) if hasattr(value, "to") else value
    return moved_inputs


def _require_torch() -> Any:
    if torch is None:
        raise ImportError(
            "Real Depth Anything V2 inference requires PyTorch. "
            "Install project dependencies with: pip install -r requirements.txt"
        )
    return torch


def _as_2d_float32_depth_map(depth_map: Any) -> np.ndarray:
    """Convert model output into a validated 2D float32 NumPy array."""
    if torch is not None and isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().float().cpu().numpy()
    else:
        depth = np.asarray(depth_map, dtype=np.float32)

    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError("Depth map must be 2D after squeezing, got shape {}".format(depth.shape))

    if not np.isfinite(depth).all():
        raise ValueError("Depth map contains NaN or infinite values.")

    return depth.astype(np.float32, copy=False)


def _resize_depth_map(depth_map: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resize a 2D depth map to the input image dimensions using bilinear interpolation."""
    if depth_map.shape == (target_height, target_width):
        return depth_map.astype(np.float32, copy=False)

    if torch is None:
        resized = Image.fromarray(depth_map.astype(np.float32), mode="F").resize(
            (target_width, target_height),
            resample=Image.BILINEAR,
        )
        return np.asarray(resized, dtype=np.float32)

    depth_tensor = torch.from_numpy(depth_map.astype(np.float32))[None, None, :, :]
    resized = torch.nn.functional.interpolate(
        depth_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].cpu().numpy().astype(np.float32, copy=False)


def _finalize_depth_map(depth_map: Any, target_height: int, target_width: int) -> np.ndarray:
    """
    Convert, resize, and validate a relative depth map.

    This is the final guard for the Depth -> Mask contract: downstream masking
    code must receive a 2D ``float32`` array whose shape exactly matches the
    original full input image.
    """
    finalized = _as_2d_float32_depth_map(depth_map)
    finalized = _resize_depth_map(finalized, target_height, target_width)

    expected_shape = (target_height, target_width)
    if finalized.shape != expected_shape:
        raise ValueError(
            "Depth map shape mismatch after resizing: expected {}, got {}".format(
                expected_shape, finalized.shape
            )
        )

    if finalized.dtype != np.float32:
        finalized = finalized.astype(np.float32, copy=False)

    return finalized


@dataclass
class DepthAnythingV2Adapter:
    """
    Thin wrapper around a Hugging Face Depth Anything V2 model and processor.

    ``estimate`` always runs inference on the full image and post-processes the
    output back to the original image size.
    """

    image_processor: Any
    model: Any
    device: str
    model_name: str = DEFAULT_DEPTH_ANYTHING_V2_MODEL

    def estimate(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        torch_module = _require_torch()

        # Depth -> Mask: run the model on the complete original image, not on
        # the drone crop, so the monocular model keeps full scene context.
        pil_image = _to_pil_rgb_image(image)
        target_height, target_width = pil_image.height, pil_image.width

        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = _tensor_dict_to_device(inputs, self.device)

        with torch_module.inference_mode():
            outputs = self.model(**inputs)

        if hasattr(self.image_processor, "post_process_depth_estimation"):
            processed = self.image_processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(target_height, target_width)],
            )
            predicted_depth = processed[0]["predicted_depth"]
        else:
            predicted_depth = outputs.predicted_depth

        return _finalize_depth_map(predicted_depth, target_height, target_width)


def load_depth_anything_v2(
    model_name: str = DEFAULT_DEPTH_ANYTHING_V2_MODEL,
    device: Optional[str] = None,
    torch_dtype: Optional[Any] = None,
    local_files_only: bool = False,
) -> DepthAnythingV2Adapter:
    """
    Load a Depth Anything V2 model using Hugging Face Transformers.

    Parameters
    ----------
    model_name
        Hugging Face model identifier. The default uses the small relative-depth
        model to keep local inference practical.
    device
        Optional PyTorch device string. Defaults to ``"cuda"`` when available,
        otherwise ``"cpu"``.
    torch_dtype
        Optional dtype for model weights, e.g. ``torch.float16`` on CUDA.

    Returns
    -------
    DepthAnythingV2Adapter
        Ready-to-use adapter for ``estimate_relative_depth``.
    """
    torch_module = _require_torch()

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise ImportError(
            "Depth Anything V2 inference requires the 'transformers' package. "
            "Install it with: pip install transformers"
        ) from exc

    resolved_device = device or ("cuda" if torch_module.cuda.is_available() else "cpu")
    model_kwargs = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    image_processor = AutoImageProcessor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        **model_kwargs
    )
    model.to(resolved_device)
    model.eval()

    return DepthAnythingV2Adapter(
        image_processor=image_processor,
        model=model,
        device=resolved_device,
        model_name=model_name,
    )


def estimate_relative_depth(image, depth_model):
    """
    Estimate a full-image relative depth map from an RGB image.

    Parameters
    ----------
    image
        Input image as a PIL image, NumPy array, or image path.
    depth_model
        Preferably a ``DepthAnythingV2Adapter`` returned by
        ``load_depth_anything_v2``. A generic callable is also accepted for
        backwards compatibility, but it must return a depth-like array.

    Returns
    -------
    numpy.ndarray
        2D relative depth map with dtype ``float32`` and the same ``H x W`` as
        the input image.
    """
    if depth_model is None:
        raise ValueError("depth_model must be provided for real depth estimation.")

    target_height, target_width = _target_size_from_image(image)

    if hasattr(depth_model, "estimate"):
        depth_map = depth_model.estimate(image)
    elif callable(depth_model):
        raw_output = depth_model(image)
        if isinstance(raw_output, dict):
            raw_output = raw_output.get("predicted_depth", raw_output.get("depth"))
            if raw_output is None:
                raise ValueError("Depth model output dict must contain 'predicted_depth' or 'depth'.")
        depth_map = raw_output
    else:
        raise TypeError("depth_model must be a DepthAnythingV2Adapter or a callable model.")

    return _finalize_depth_map(depth_map, target_height, target_width)


def estimate_relative_depth_demo(image, width=640, height=480):
    """
    Demo relative depth map for baseline testing.
    """
    import numpy as np

    depth = np.linspace(0.5, 2.5, height * width)
    return depth.reshape((height, width)).astype(np.float32)

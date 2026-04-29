"""
Dataset loader for the drone monocular-depth dataset.

Current strict dataset contract:

1. Only ``.png`` images are considered.
2. Metadata is encoded in the image filename, not in the folder structure.
3. Each matching YOLO ``.txt`` file contains exactly one annotation line.

Expected filename pattern:

    HighresScreenshot00001_depth_40_light_rain_10AM.png

From this example the loader extracts:

    true_distance_m = 40.0
    weather = "light_rain"
    time_of_day = "10AM"

The matching YOLO label file must have the same base name:

    HighresScreenshot00001_depth_40_light_rain_10AM.txt

YOLO label format:

    class_id x_center y_center width height

Coordinates are normalized to [0, 1] and are converted to absolute pixel
coordinates in ``(x1, y1, x2, y2)`` format.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple, Union

from PIL import Image, UnidentifiedImageError


LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path(r"C:\Users\depthlev\Desktop\droneImages\dataset")
IMAGE_EXTENSION = ".png"


@dataclass(frozen=True)
class YoloAnnotation:
    """Single object annotation parsed from one YOLO label row."""

    class_id: int
    bbox: Tuple[int, int, int, int]
    normalized_bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class DroneDatasetSample:
    """One usable image sample with filename metadata and one YOLO annotation."""

    image_path: Path
    label_path: Path
    true_distance_m: float
    weather: str
    time_of_day: str
    image_width: int
    image_height: int
    annotation: YoloAnnotation

    @property
    def annotations(self) -> List[YoloAnnotation]:
        """
        Backwards-compatible one-item annotation list.

        New code should prefer ``sample.annotation`` because the dataset contract
        guarantees exactly one YOLO row per image.
        """
        return [self.annotation]


def _coerce_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _read_image_size(image_path: Path) -> Tuple[int, int]:
    """
    Read image dimensions and verify that the file is decodable.

    Returns
    -------
    tuple
        ``(width, height)`` in pixels.
    """
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Unreadable image file: {}".format(image_path)) from exc

    if width <= 0 or height <= 0:
        raise ValueError("Image has invalid dimensions: {}".format(image_path))

    return width, height


def load_image_rgb(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image as a PIL RGB image.

    The returned image owns its pixel data, so the source file can be closed
    immediately after this function returns.
    """
    path = _coerce_path(image_path)
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Unable to load image: {}".format(path)) from exc


def extract_metadata_from_filename(image_path: Union[str, Path]) -> Tuple[float, str, str]:
    """
    Extract ``(true_distance_m, weather, time_of_day)`` from the image filename.

    The parser splits the filename stem by underscores and looks for the
    ``depth`` token. The token after ``depth`` is parsed as the metric distance,
    the final token is parsed as the time of day, and all tokens between the
    distance and final token are joined back into the weather string.

    Example
    -------
    ``HighresScreenshot00001_depth_40_light_rain_10AM.png`` becomes
    ``(40.0, "light_rain", "10AM")``.
    """
    path = _coerce_path(image_path)
    if path.suffix.lower() != IMAGE_EXTENSION:
        raise ValueError("Expected a .png image filename: {}".format(path.name))

    parts = path.stem.split("_")
    try:
        depth_index = parts.index("depth")
    except ValueError as exc:
        raise ValueError("Filename does not contain '_depth_': {}".format(path.name)) from exc

    if depth_index + 3 >= len(parts):
        raise ValueError(
            "Filename must match '<prefix>_depth_<distance>_<weather>_<time>.png': {}".format(
                path.name
            )
        )

    distance_token = parts[depth_index + 1]
    weather_tokens = parts[depth_index + 2 : -1]
    time_of_day = parts[-1]

    if not weather_tokens:
        raise ValueError("Filename is missing weather metadata: {}".format(path.name))

    try:
        true_distance_m = float(distance_token)
    except ValueError as exc:
        raise ValueError(
            "Filename depth value is not numeric in '{}': {}".format(
                path.name, distance_token
            )
        ) from exc

    weather = "_".join(weather_tokens)
    return true_distance_m, weather, time_of_day


# Backwards-compatible alias for older imports.
find_depth_metadata = extract_metadata_from_filename


def yolo_normalized_to_absolute_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Convert a normalized YOLO bbox into absolute pixel coordinates.

    Returned ``x2`` and ``y2`` are exclusive slice endpoints, matching
    ``depth_map[y1:y2, x1:x2]`` in the Depth -> Mask stage.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if width <= 0.0 or height <= 0.0:
        raise ValueError("YOLO bbox width and height must be positive.")

    values = (x_center, y_center, width, height)
    if any(value < -1e-6 or value > 1.0 + 1e-6 for value in values):
        raise ValueError("YOLO bbox values must be normalized to [0, 1].")

    x1_float = (x_center - width / 2.0) * image_width
    y1_float = (y_center - height / 2.0) * image_height
    x2_float = (x_center + width / 2.0) * image_width
    y2_float = (y_center + height / 2.0) * image_height

    x1 = max(0, min(int(round(x1_float)), image_width - 1))
    y1 = max(0, min(int(round(y1_float)), image_height - 1))
    x2 = max(0, min(int(round(x2_float)), image_width))
    y2 = max(0, min(int(round(y2_float)), image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Converted YOLO bbox is empty after clamping.")

    return x1, y1, x2, y2


def parse_yolo_label_file(
    label_path: Union[str, Path],
    image_width: int,
    image_height: int,
) -> YoloAnnotation:
    """
    Parse the single YOLO annotation row for one image.

    The label file is expected to contain exactly one non-empty line:

        class_id x_center y_center width height
    """
    path = _coerce_path(label_path)

    try:
        line = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError("Unable to read YOLO label file: {}".format(path)) from exc

    if not line:
        raise ValueError("YOLO label file is empty: {}".format(path))

    columns = line.split()
    if len(columns) != 5:
        raise ValueError(
            "{} must contain exactly one YOLO row with 5 columns; got {}".format(
                path, len(columns)
            )
        )

    try:
        class_id = int(float(columns[0]))
        x_center, y_center, width, height = [float(value) for value in columns[1:]]
        bbox = yolo_normalized_to_absolute_bbox(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
        )
    except ValueError as exc:
        raise ValueError("Invalid YOLO annotation in {}".format(path)) from exc

    return YoloAnnotation(
        class_id=class_id,
        bbox=bbox,
        normalized_bbox=(x_center, y_center, width, height),
    )


class DroneDatasetLoader:
    """
    Traverse a dataset root and yield validated ``.png`` samples.

    Invalid samples are skipped by default. Set ``strict=True`` to fail fast on
    missing labels, unreadable images, malformed filenames, or malformed labels.
    """

    def __init__(
        self,
        dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
        strict: bool = False,
    ) -> None:
        self.dataset_root = _coerce_path(dataset_root)
        self.strict = strict

    def iter_image_paths(self) -> Iterator[Path]:
        """Yield only ``.png`` image paths under ``dataset_root``."""
        if not self.dataset_root.exists():
            raise FileNotFoundError("Dataset root does not exist: {}".format(self.dataset_root))

        if not self.dataset_root.is_dir():
            raise NotADirectoryError(
                "Dataset root is not a directory: {}".format(self.dataset_root)
            )

        for path in sorted(self.dataset_root.rglob("*{}".format(IMAGE_EXTENSION))):
            if path.is_file() and path.suffix.lower() == IMAGE_EXTENSION:
                yield path

    def iter_samples(self) -> Iterator[DroneDatasetSample]:
        """Yield usable samples with filename metadata and one parsed annotation."""
        for image_path in self.iter_image_paths():
            label_path = image_path.with_suffix(".txt")

            if not label_path.exists():
                message = "Missing YOLO label for image: {}".format(image_path)
                if self.strict:
                    raise FileNotFoundError(message)
                LOGGER.warning("%s; skipping image", message)
                continue

            try:
                image_width, image_height = _read_image_size(image_path)
                true_distance_m, weather, time_of_day = extract_metadata_from_filename(
                    image_path
                )
                annotation = parse_yolo_label_file(
                    label_path=label_path,
                    image_width=image_width,
                    image_height=image_height,
                )
            except (ValueError, OSError) as exc:
                if self.strict:
                    raise
                LOGGER.warning("Skipping image %s: %s", image_path, exc)
                continue

            yield DroneDatasetSample(
                image_path=image_path,
                label_path=label_path,
                true_distance_m=true_distance_m,
                weather=weather,
                time_of_day=time_of_day,
                image_width=image_width,
                image_height=image_height,
                annotation=annotation,
            )

    def load_samples(self) -> List[DroneDatasetSample]:
        """Return all usable samples as a list."""
        return list(self.iter_samples())

    @staticmethod
    def load_image(sample: DroneDatasetSample) -> Image.Image:
        """Load a sample image as RGB."""
        return load_image_rgb(sample.image_path)


def load_drone_dataset(
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
    strict: bool = False,
) -> List[DroneDatasetSample]:
    """Convenience function that returns all usable samples from a dataset root."""
    return DroneDatasetLoader(dataset_root=dataset_root, strict=strict).load_samples()

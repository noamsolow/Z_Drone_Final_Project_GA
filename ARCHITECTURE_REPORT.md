# Monocular Drone Detection and 3D Localization - Current Pipeline Architecture Report

Generated from the current repository state on 2026-04-23.

## 1. Executive Summary

The current codebase contains a lightweight, modular baseline pipeline for monocular drone 3D localization. The executable pipeline is concentrated under `pipeline/` and is organized around two modes:

- Calibration mode: estimates a scalar conversion factor from relative monocular depth to metric Z-distance.
- Runtime mode: detects a drone, estimates relative depth, converts it to metric depth, and reconstructs a camera-centric 3D position.

The current implementation is best described as a scaffold with several working mathematical utility functions and several placeholder/demo components. The demo path runs end-to-end using synthetic/random data. The real-model path is only partially implemented because it expects externally loaded detector and depth models and does not include model loading, data loading, training, evaluation, camera calibration, or class-specific YOLO filtering.

## 2. Current High-Level Architecture and Pipeline Flow

### 2.1 Calibration Flow

Current implementation: `pipeline.runner.run_calibration_pipeline`

```text
Input images + known true distances
        |
        v
Object detection
  - demo: random bounding box
  - real: caller-provided YOLO-like model
        |
        v
Relative depth estimation
  - demo: synthetic gradient depth map
  - real: caller-provided depth model
        |
        v
Extract drone relative depth from detected bounding box
        |
        v
Fit scale factor s using least squares:
Z_true ~= s * d_rel
        |
        v
Return scale factor s
```

Purpose: The calibration flow estimates a single global scale factor `s` that converts unitless relative depth values into approximate metric depth values.

Current limitations:

- There is no dataset loader.
- There is no camera calibration input file.
- There is no train/validation split.
- There is no persistence of the fitted scale factor.
- The default demo path uses random detections and synthetic depth maps, so repeated runs can produce different scale factors.
- The real path requires the caller to provide already-loaded models.

### 2.2 Runtime Inference Flow

Current implementation: `pipeline.runner.run_runtime_pipeline`

```text
Input RGB image
        |
        v
Object detection
  - output bbox and bbox center pixel
        |
        v
Relative depth estimation
  - output H x W relative depth map
        |
        v
Extract representative depth inside drone bbox
        |
        v
Metric conversion:
z_hat = scale_factor * d_rel
        |
        v
Inverse camera projection:
X = (x_pixel - cx) * Z / fx
Y = (y_pixel - cy) * Z / fy
Z = z_hat
        |
        v
Return camera-centric 3D position: (X, Y, Z)
```

Purpose: The runtime flow estimates the drone position in the camera coordinate frame from a single image.

Current limitations:

- There is no image/video ingestion wrapper.
- There is no real detector loading.
- There is no real depth-model loading or preprocessing.
- There is no drone class filtering in the detector wrapper.
- There is no output confidence, uncertainty estimate, or evaluation metric.
- There is no temporal smoothing or multi-frame tracking.
- There is no conversion from camera coordinates to world coordinates.

### 2.3 Expected Full Pipeline Versus Current Code

| Stage | Expected Full Project Stage | Current Implementation |
| --- | --- | --- |
| Data loading | Load images, annotations, ground-truth distances, camera intrinsics | Not implemented. Caller passes image objects and true distances directly. |
| Object detection | Train/load YOLO drone detector, detect drone class, return bbox/score | Partially implemented. Real wrapper calls provided YOLO-like model; demo returns random bbox. |
| Depth estimation | Load pretrained or trained monocular depth model, preprocess image, infer relative depth | Partially implemented. Real wrapper calls provided model; demo returns synthetic gradient. |
| Depth extraction | Select drone pixels/region and compute representative relative depth | Implemented with median over bbox region. |
| Metric conversion | Convert relative depth to metric depth using calibration | Implemented inline in runtime as `z_hat = scale_factor * d_rel`; calibration scale fitting implemented. |
| 3D localization | Use camera intrinsics and metric depth to compute `(X, Y, Z)` | Implemented using pinhole inverse projection. |
| Evaluation | Compare predictions to ground truth and report metrics | Not implemented. |
| Training | Train/fine-tune detector or depth model | Not implemented. |

## 3. Repository Structure Relevant to the Pipeline

```text
pipeline/
  __init__.py
  runner.py
  detection/
    __init__.py
    detector.py
  depth/
    __init__.py
    depth_model.py
    drone_depth.py
  calibration/
    __init__.py
    scale_calibration.py
  geometry/
    __init__.py
    projection.py

experiments/
  background_drone_exp/
  bb_on_drones_exp/
  bounding_box_exp/
  depth_models_exp/

README.md
requirements.txt
```

The `experiments/` directories contain notebooks, images, and generated outputs. They are useful project artifacts but are not currently connected to the importable `pipeline/` package as reusable modules.

## 4. Modules and Classes

### 4.1 Core Pipeline Modules

| File | Role | Main Contents | Implementation Status |
| --- | --- | --- | --- |
| `pipeline/runner.py` | Top-level orchestration for calibration and runtime inference | `run_calibration_pipeline`, `run_runtime_pipeline`, demo `__main__` block | Partially implemented orchestration. Demo path works; real path depends on externally supplied models. |
| `pipeline/detection/detector.py` | Drone detection abstraction | `detect_drone`, `detect_drone_demo` | Real YOLO-style wrapper is partial; demo detector is placeholder/random. |
| `pipeline/depth/depth_model.py` | Relative depth model abstraction | `estimate_relative_depth`, `estimate_relative_depth_demo` | Real wrapper is generic and partial; demo depth is synthetic placeholder. |
| `pipeline/depth/drone_depth.py` | Extract drone depth value from depth map and bbox | `extract_drone_relative_depth` | Implemented numerical logic. |
| `pipeline/calibration/scale_calibration.py` | Estimate relative-to-metric scale factor | `compute_scale_factor` | Implemented least-squares logic. |
| `pipeline/geometry/projection.py` | Convert 2D pixel + depth into 3D camera coordinates | `reconstruct_drone_position` | Implemented pinhole camera inverse projection. |

### 4.2 Package Initializers

| File | Role | Implementation Status |
| --- | --- | --- |
| `pipeline/__init__.py` | Package-level docstring only | No runtime logic. |
| `pipeline/detection/__init__.py` | Detection subpackage docstring only | No runtime logic. |
| `pipeline/depth/__init__.py` | Depth subpackage docstring only | No runtime logic. |
| `pipeline/calibration/__init__.py` | Calibration subpackage docstring only | No runtime logic. |
| `pipeline/geometry/__init__.py` | Geometry subpackage docstring only | No runtime logic. |

### 4.3 Classes

There are currently no Python classes defined in the `pipeline/` package. The architecture is currently procedural and function-based.

## 5. Function-by-Function Breakdown

### 5.1 `pipeline.runner.run_calibration_pipeline`

Source: `pipeline/runner.py`

Signature:

```python
def run_calibration_pipeline(
    images,
    true_distances,
    detection_mode="demo",
    depth_mode="demo",
    detector_model=None,
    depth_model=None,
):
```

Inputs:

| Argument | Expected Type / Shape | Description |
| --- | --- | --- |
| `images` | Iterable of RGB images | Calibration images. In demo mode these may be `None` because demo functions ignore image content. |
| `true_distances` | Iterable of numeric values | Ground-truth drone Z-distances in meters. |
| `detection_mode` | String, expected `"demo"` or `"real"` | Selects random demo detector or real detector wrapper. |
| `depth_mode` | String, expected `"demo"` or `"real"` | Selects synthetic demo depth or real depth wrapper. |
| `detector_model` | Model object or `None` | Required by real detection path. Expected to behave like an Ultralytics YOLO model. |
| `depth_model` | Model object or `None` | Required by real depth path. Expected to be callable on an image. |

Outputs:

| Return | Type | Description |
| --- | --- | --- |
| `s` | `float` | Estimated scalar scale factor mapping relative depth to metric Z-distance. |

Description:

Runs the offline calibration stage. For each `(image, z_true)` pair, it detects the drone, estimates a relative depth map, extracts a representative relative depth inside the detected drone bounding box, accumulates `(d_rel, z_true)` pairs, and calls `compute_scale_factor` to fit `Z_true ~= s * d_rel`.

Implementation status:

Partially implemented orchestration. The control flow and scale fitting are real, but the default mode uses random/synthetic demo data. The real mode delegates to externally supplied models and does not load, validate, or configure those models. There is no validation that `detection_mode` or `depth_mode` are valid strings; any non-`"demo"` value falls through to the real path. There is no check that `images` and `true_distances` have equal length beyond the truncation behavior of `zip`.

Important dependencies:

| Dependency | Function |
| --- | --- |
| `detect_drone_demo` or `detect_drone` | Produces the drone bbox and center pixel. |
| `estimate_relative_depth_demo` or `estimate_relative_depth` | Produces the depth map. |
| `extract_drone_relative_depth` | Reduces bbox depth values to a scalar. |
| `compute_scale_factor` | Fits the final scale factor. |

### 5.2 `pipeline.runner.run_runtime_pipeline`

Source: `pipeline/runner.py`

Signature:

```python
def run_runtime_pipeline(
    image,
    scale_factor,
    intrinsics,
    detection_mode="demo",
    depth_mode="demo",
    detector_model=None,
    depth_model=None,
):
```

Inputs:

| Argument | Expected Type / Shape | Description |
| --- | --- | --- |
| `image` | RGB image | Runtime image containing a drone. |
| `scale_factor` | Numeric scalar | Precomputed calibration factor `s`. |
| `intrinsics` | Dict with `fx`, `fy`, `cx`, `cy` | Camera intrinsic parameters. |
| `detection_mode` | String, expected `"demo"` or `"real"` | Selects demo or real detector. |
| `depth_mode` | String, expected `"demo"` or `"real"` | Selects demo or real depth model. |
| `detector_model` | Model object or `None` | Required by real detection path. |
| `depth_model` | Model object or `None` | Required by real depth path. |

Outputs:

| Return | Type | Description |
| --- | --- | --- |
| `position_3d` | Tuple `(X, Y, Z)` | Estimated drone position in camera coordinates. |

Description:

Runs single-image inference. It detects the drone, estimates a relative depth map, extracts relative depth inside the bounding box, converts relative depth to metric depth using `scale_factor`, and reconstructs camera-frame 3D coordinates with `reconstruct_drone_position`.

Implementation status:

Partially implemented orchestration. The math path is real once valid detection, depth map, scale factor, and intrinsics are provided. The default mode still uses random/synthetic placeholder outputs. The function has no image loading, no model loading, no confidence propagation, no evaluation output, and no validation for unsupported mode strings. It assumes the detector result contains `"bbox"`, `"x_pixel"`, and `"y_pixel"`.

Important dependencies:

| Dependency | Function |
| --- | --- |
| `detect_drone_demo` or `detect_drone` | Produces bbox and center pixel. |
| `estimate_relative_depth_demo` or `estimate_relative_depth` | Produces depth map. |
| `extract_drone_relative_depth` | Produces scalar relative depth. |
| `reconstruct_drone_position` | Produces 3D camera coordinates. |

### 5.3 `pipeline.detection.detector.detect_drone`

Source: `pipeline/detection/detector.py`

Signature:

```python
def detect_drone(image, model, conf_threshold=0.25):
```

Inputs:

| Argument | Expected Type / Shape | Description |
| --- | --- | --- |
| `image` | RGB image or model-compatible image input | Image passed directly into the detector model. |
| `model` | YOLO-like callable | Expected to support `model(image, conf=conf_threshold)`. |
| `conf_threshold` | `float`, default `0.25` | Detector confidence threshold passed into the model. |

Outputs:

| Return | Type | Description |
| --- | --- | --- |
| Detection dict | `dict` | Contains `"bbox": (x1, y1, x2, y2)`, `"x_pixel": x_center`, and `"y_pixel": y_center`. |

Raises:

| Exception | Condition |
| --- | --- |
| `RuntimeError` | Raised if the model returns no boxes. |

Description:

Calls a YOLO-like detector on a single image, reads `results[0].boxes`, selects the highest-confidence box, converts the bounding box from `xyxy` tensor format to Python values, computes the center pixel, and returns the standardized detection dictionary used by the rest of the pipeline.

Implementation status:

Partially implemented real-model adapter. It contains actual detection-result parsing logic for a compatible Ultralytics-style YOLO model, but the codebase does not include model loading, class-name mapping, checkpoint paths, or training logic. It also does not filter detections by drone class; it simply selects the highest-confidence detection among all boxes returned by the model. This is not yet production-ready for drone-only localization unless the provided model returns only drone detections or class filtering is added.

Key implementation details:

| Detail | Current Behavior |
| --- | --- |
| Model call | `results = model(image, conf=conf_threshold)` |
| Box source | `results[0].boxes` |
| Selection policy | Highest confidence only |
| Box format | Assumes `best_box.xyxy[0].tolist()` |
| Returned bbox type | Integer `(x1, y1, x2, y2)` |
| Returned center type | Float `x_center`, `y_center` |

### 5.4 `pipeline.detection.detector.detect_drone_demo`

Source: `pipeline/detection/detector.py`

Signature:

```python
def detect_drone_demo(image, image_width=640, image_height=480):
```

Inputs:

| Argument | Expected Type / Shape | Description |
| --- | --- | --- |
| `image` | Any value | Unused placeholder argument, kept for interface compatibility. |
| `image_width` | `int`, default `640` | Synthetic image width used to generate bbox coordinates. |
| `image_height` | `int`, default `480` | Synthetic image height used to generate bbox coordinates. |

Outputs:

| Return | Type | Description |
| --- | --- | --- |
| Detection dict | `dict` | Contains random `"bbox"` and random center `"x_pixel"`, `"y_pixel"`. |

Description:

Generates a random bounding box in a plausible upper/middle image region. The width and height are random fractions of the provided image dimensions. The box is clamped to the synthetic image bounds.

Implementation status:

Placeholder/demo logic. It does not inspect the input image and does not perform actual detection. It is stochastic and no random seed is set, so outputs vary between calls and runs. It exists to keep the pipeline runnable before a real detector is integrated.

Key implementation details:

| Detail | Current Behavior |
| --- | --- |
| Uses image content | No |
| Box width | Random `4%` to `12%` of `image_width` |
| Box height | Random `3%` to `10%` of `image_height` |
| Center x | Random `25%` to `75%` of image width |
| Center y | Random `20%` to `60%` of image height |
| Bounds handling | Clamps bbox to `[0, width - 1]` and `[0, height - 1]` |

### 5.5 `pipeline.depth.depth_model.estimate_relative_depth`

Source: `pipeline/depth/depth_model.py`

Signature:

```python
def estimate_relative_depth(image, depth_model):
```

Inputs:

| Argument | Expected Type / Shape | Description |
| --- | --- | --- |
| `image` | RGB image or model-compatible image input | Image passed directly into the depth model. |
| `depth_model` | Callable model object | Expected to return a depth-like array when called with `image`. |

Outputs:

| Return | Type / Shape | Description |
| --- | --- | --- |
| `depth_map` | `numpy.ndarray`, shape expected `H x W`, dtype `float32` | Relative depth map. Unitless. |

Description:

Calls the provided depth model on the image and converts the result to a NumPy `float32` array. The module docstring says this is for Depth Anything, but the function is generic and will accept any callable that returns an array-like depth map.

Implementation status:

Partially implemented real-model adapter. It contains real wrapper logic for a preloaded callable model, but there is no Depth Anything, MiDaS, or ZoeDepth model loader; no preprocessing; no device placement; no resizing; no normalization; no output orientation convention checks; and no postprocessing. It assumes the model output can be directly interpreted as an `H x W` relative depth map.

Key implementation details:

| Detail | Current Behavior |
| --- | --- |
| Model loading | Not implemented |
| Preprocessing | Not implemented |
| Inference call | `depth_model(image)` |
| Output conversion | `np.asarray(depth_map, dtype=np.float32)` |
| Shape validation | Not implemented |

### 5.6 `pipeline.depth.depth_model.estimate_relative_depth_demo`

Source: `pipeline/depth/depth_model.py`

Signature:

```python
def estimate_relative_depth_demo(image, width=640, height=480):
```

Inputs:

| Argument | Expected Type / Shape | Description |
| --- | --- | --- |
| `image` | Any value | Unused placeholder argument. |
| `width` | `int`, default `640` | Width of generated depth map. |
| `height` | `int`, default `480` | Height of generated depth map. |

Outputs:

| Return | Type / Shape | Description |
| --- | --- | --- |
| Demo depth map | `numpy.ndarray`, shape `(height, width)`, dtype `float32` | Synthetic relative depth gradient from `0.5` to `2.5`. |

Description:

Creates a linearly increasing synthetic depth map using `np.linspace(0.5, 2.5, height * width)`, reshapes it to `(height, width)`, and returns it as `float32`.

Implementation status:

Placeholder/demo logic. It does not inspect the input image and does not estimate real scene depth. It exists only to exercise downstream depth extraction, calibration, and projection code.

Key implementation details:

| Detail | Current Behavior |
| --- | --- |
| Uses image content | No |
| Min value | `0.5` |
| Max value | `2.5` |
| Spatial pattern | Linear raster-order gradient |
| Deterministic | Yes, for fixed width and height |

### 5.7 `pipeline.depth.drone_depth.extract_drone_relative_depth`

Source: `pipeline/depth/drone_depth.py`

Signature:

```python
def extract_drone_relative_depth(depth_map, bbox):
```

Inputs:

| Argument | Expected Type / Shape | Description |
| --- | --- | --- |
| `depth_map` | Array-like, expected shape `H x W` | Relative depth map from the depth model. |
| `bbox` | Tuple `(x1, y1, x2, y2)` | Drone bounding box in pixel coordinates. |

Outputs:

| Return | Type | Description |
| --- | --- | --- |
| `relative_depth` | `float` | Median relative depth value inside the detected drone bounding box. |

Raises:

| Exception | Condition |
| --- | --- |
| `ValueError` | Raised if the clamped bbox is invalid. |
| `ValueError` | Raised if the bbox slice contains no depth values. |

Description:

Clamps the input bounding box to the depth map dimensions, slices the depth map inside the box, computes the median of the selected depth values, and returns that scalar as the representative drone relative depth.

Implementation status:

Implemented working numerical logic. This function is one of the more complete parts of the current pipeline. It handles out-of-bounds boxes through clamping and guards against empty/invalid regions.

Key implementation details:

| Detail | Current Behavior |
| --- | --- |
| Bounds handling | Clamps bbox to depth map width/height |
| Region selection | `depth_map[y1:y2, x1:x2]` |
| Aggregation | Median using `np.median` |
| Noise robustness | Median is more robust than mean for outliers |
| Mask support | Not implemented; bbox region only |

Potential future improvement:

Use a drone segmentation mask, center-weighted region, or percentile filtering to avoid background pixels inside the bbox.

### 5.8 `pipeline.calibration.scale_calibration.compute_scale_factor`

Source: `pipeline/calibration/scale_calibration.py`

Signature:

```python
def compute_scale_factor(relative_depths, true_distances):
```

Inputs:

| Argument | Expected Type / Shape | Description |
| --- | --- | --- |
| `relative_depths` | Sequence of numeric values | Relative depth values `d_rel` extracted from calibration images. |
| `true_distances` | Sequence of numeric values | Ground-truth metric Z-distances `Z_true`, in meters. |

Outputs:

| Return | Type | Description |
| --- | --- | --- |
| `s` | `float` | Least-squares scale factor such that `Z_true ~= s * d_rel`. |

Raises:

| Exception | Condition |
| --- | --- |
| `ValueError` | Raised if either sequence is empty. |
| `ValueError` | Raised if sequence lengths differ. |
| `ValueError` | Raised if all relative depth values are zero. |

Description:

Converts inputs to NumPy `float32` arrays and solves a one-parameter linear regression through the origin:

```text
s = sum(d_rel * Z_true) / sum(d_rel * d_rel)
```

Implementation status:

Implemented working calibration math. The function is numerically simple and includes basic validation. It assumes that a single global scale factor is sufficient for the camera/model setup and that relative depth increases in the same direction as true distance.

Key implementation details:

| Detail | Current Behavior |
| --- | --- |
| Regression type | Least squares through origin |
| Uses intercept | No |
| Input dtype | Converts to `np.float32` |
| Empty input guard | Yes |
| Length mismatch guard | Yes |
| Zero denominator guard | Yes |

Potential future improvement:

Consider robust regression, per-scene calibration, validation metrics, confidence intervals, and explicit handling of inverse-depth models where larger relative values may mean closer rather than farther.

### 5.9 `pipeline.geometry.projection.reconstruct_drone_position`

Source: `pipeline/geometry/projection.py`

Signature:

```python
def reconstruct_drone_position(x_pixel, y_pixel, z, intrinsics):
```

Inputs:

| Argument | Expected Type / Shape | Description |
| --- | --- | --- |
| `x_pixel` | Numeric scalar | Drone center x-coordinate in image pixels. |
| `y_pixel` | Numeric scalar | Drone center y-coordinate in image pixels. |
| `z` | Numeric scalar | Estimated metric depth in meters. |
| `intrinsics` | Dict with `fx`, `fy`, `cx`, `cy` | Camera intrinsic parameters. |

Outputs:

| Return | Type | Description |
| --- | --- | --- |
| `(X, Y, Z)` | Tuple of numeric values | Drone position in camera coordinates. |

Raises:

| Exception | Condition |
| --- | --- |
| `ValueError` | Raised if `fx` or `fy` equals zero. |
| `KeyError` | Implicitly raised if required intrinsic keys are missing. |

Description:

Performs inverse projection from pixel coordinates into camera-centric 3D coordinates using a pinhole camera model:

```text
X = (x_pixel - cx) * z / fx
Y = (y_pixel - cy) * z / fy
Z = z
```

Implementation status:

Implemented working geometric math. This function assumes valid intrinsics and that `z` is already in metric units. It returns camera-frame coordinates only; no camera pose, extrinsics, or world-frame conversion are implemented.

Key implementation details:

| Detail | Current Behavior |
| --- | --- |
| Projection model | Pinhole inverse projection |
| Coordinate frame | Camera-centric |
| Distortion handling | Not implemented |
| Extrinsics/world conversion | Not implemented |
| Intrinsic validation | Only checks `fx` and `fy` are non-zero |

## 6. Script Entry Point

### `python -m pipeline.runner`

Source: `pipeline/runner.py`

Behavior:

When executed as a module, `pipeline.runner` creates:

- `num_samples = 10`
- `fake_images = [None] * num_samples`
- Synthetic true distances: `[20, 25, 30, 35, 40, 45, 50, 55, 60, 65]`

It then calls:

```python
run_calibration_pipeline(
    images=fake_images,
    true_distances=true_distances,
    detection_mode="demo",
    depth_mode="demo",
)
```

and prints the estimated scale factor.

Implementation status:

Demo smoke test only. It verifies that calibration mode can execute with placeholder data. It does not run runtime 3D localization, does not load real images, and does not evaluate accuracy.

Verification result:

The command `python -m pipeline.runner` executed successfully in the current environment and printed an estimated scale factor. Because demo detection is random, the exact value is not stable between runs.

## 7. Current Data and Experiment Assets

The repository contains several experiment directories:

| Directory | Observed Contents | Pipeline Integration Status |
| --- | --- | --- |
| `experiments/bb_on_drones_exp/` | Drone images and notebook | Not imported by `pipeline/`. |
| `experiments/bounding_box_exp/` | Input images, masked images, generated depth maps, notebook | Not imported by `pipeline/`. |
| `experiments/depth_models_exp/` | Depth model comparison notebook and outputs | Not imported by `pipeline/`. |
| `experiments/background_drone_exp/` | Background/depth comparison notebook, input images, generated outputs | Not imported by `pipeline/`. |

These assets appear to support exploratory work, but the current production pipeline does not consume them through a formal dataset API.

## 8. Implementation Status Summary

| Component | Status | Notes |
| --- | --- | --- |
| Pipeline package layout | Implemented | Clean modular separation by detection, depth, calibration, and geometry. |
| Calibration orchestrator | Partial | Works with demo data; real model path requires external models. |
| Runtime orchestrator | Partial | Core sequence exists; no model loading or evaluation. |
| YOLO detector wrapper | Partial | Calls YOLO-like model but lacks class filtering/model setup. |
| Demo detector | Placeholder | Random bbox; ignores image. |
| Depth model wrapper | Partial | Calls arbitrary depth model but lacks model-specific preprocessing/loading. |
| Demo depth model | Placeholder | Synthetic gradient; ignores image. |
| Drone depth extraction | Implemented | Median over bbox region. |
| Scale calibration | Implemented | Least-squares through origin. |
| 3D projection | Implemented | Pinhole inverse projection in camera frame. |
| Data loading | Missing | No dataset abstraction. |
| Training | Missing | No detector/depth training scripts. |
| Evaluation | Missing | No metrics, reports, or benchmark runner. |
| Model configuration | Missing | No config files/checkpoint paths. |
| Camera calibration management | Missing | Intrinsics must be passed manually. |

## 9. Key Architectural Strengths

- The code already separates major pipeline concerns into detection, depth estimation, depth extraction, calibration, and geometry modules.
- Detection and depth functions use a common interface for demo and real paths, which makes future replacement straightforward.
- The numerical core for scale fitting and 3D projection is simple, readable, and testable.
- The pipeline can already run as a minimal smoke test with `python -m pipeline.runner`.

## 10. Key Gaps Before Actual Logic and Model Training

| Gap | Why It Matters | Suggested Next Step |
| --- | --- | --- |
| No dataset loader | Training/evaluation needs repeatable image, label, depth, distance, and intrinsic loading | Add a `pipeline/data/` module with dataset classes or simple loaders. |
| No detector training/inference setup | The YOLO wrapper cannot run without externally provided model objects | Add model loading, checkpoint configuration, class filtering, and confidence outputs. |
| No depth model setup | Real depth inference needs preprocessing, device handling, and output normalization | Add model-specific adapters for Depth Anything, MiDaS, or ZoeDepth. |
| No evaluation module | Accuracy cannot be tracked or compared | Add metrics for Z error, 3D Euclidean error, bbox IoU, and detection recall. |
| No config system | Paths, thresholds, intrinsics, and modes are hardcoded or passed manually | Add YAML/JSON config or dataclass configuration. |
| No reproducibility controls | Demo detection randomness changes calibration outputs | Add random seeds for demos and deterministic experiment settings. |
| No tests | Future implementation changes may silently break math or interfaces | Add unit tests for scale calibration, bbox depth extraction, projection, and mode dispatch. |
| No camera distortion/extrinsics | Real localization may need distortion correction and world-frame coordinates | Add camera calibration model and optional world transform. |

## 11. Recommended Near-Term Build Order

1. Add tests for the existing implemented functions: `compute_scale_factor`, `extract_drone_relative_depth`, and `reconstruct_drone_position`.
2. Add strict validation for `detection_mode`, `depth_mode`, model presence, image/depth dimensions, and calibration input lengths.
3. Add a simple config file for intrinsics, model paths, thresholds, and selected modes.
4. Implement detector model loading and drone-class filtering.
5. Implement a depth-model adapter with explicit preprocessing and output conventions.
6. Add an evaluation runner that consumes a labeled dataset and reports Z-error and 3D localization error.
7. Connect experiment assets or a formal dataset into a reusable data-loading layer.

## 12. Bottom Line

The current codebase is a well-separated pipeline skeleton with working calibration math, depth-region aggregation, and camera projection. However, the actual computer vision components are not yet fully implemented inside the repository. Detection and depth currently depend either on placeholders or externally supplied model objects. The next major milestone should be converting this scaffold into a reproducible real pipeline by adding dataset loading, model adapters, configuration, evaluation, and tests before starting serious model training.

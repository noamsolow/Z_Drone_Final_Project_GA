Here is the text cleanly formatted into proper Markdown style, with your mathematical equations updated to use standard Markdown LaTeX delimiters (`$` for inline and `$$` for block equations).

---

## VI. Proposed Method

The proposed **AirDepth** framework estimates the metric distance of a target drone from a single RGB image. The method is hybrid: it combines object detection, monocular relative-depth estimation, bounding-box geometry, scene metadata, and supervised regression. The final system does not treat the monocular depth map as metric depth directly. Instead, it learns how relative depth, drone image scale, and contextual cues map to true metric distance.

### A. Framework Overview

AirDepth consists of four main stages:

1. **Drone detection**: locate the target drone and obtain its image bounding box.
2. **Relative-depth extraction**: run a monocular depth estimator and extract depth cues from the drone region.
3. **Feature construction and regression**: combine relative depth, bounding-box geometry, and metadata into a tabular feature vector.
4. **Metric localization**: estimate the drone’s metric $Z$-distance, and optionally back-project it to a camera-relative 3D point.

```text
RGB Image It
   |
   v
Drone Detector / Annotation
   |
   v
Bounding Box bt = (ut, vt, wt, ht)
   |
   +----------------------+
   |                      |
   v                      v
BBox Geometry         Monocular Depth Model
Features             Relative Depth Map Rt
   |                      |
   +----------+-----------+
              v
      Feature Vector phi_t
              |
              v
   RF + XGBoost Distance Regressor
              |
              v
       Estimated Depth z_hat_t
              |
              v
 Optional Camera Back-Projection
       p_hat_t = (x_hat_t, y_hat_t, z_hat_t)

```

The final model was chosen through several experimental attempts. The early baseline showed that a single relative-depth value alone is weak, but combining it with bounding-box geometry and metadata improves distance estimation. Later studies tested multiple depth crops, different depth aggregations, learned depth-only random forests, noisy bounding boxes, Random Forest, XGBoost, and model blending. The strongest synthetic result came from an RF/XGBoost ensemble trained on noisy bounding-box rows, reaching approximately **7.63 m MAE** on the synthetic held-out test set.

External real-drone evaluation showed an important limitation: the uncalibrated model over-relied on bounding-box geometry and overpredicted real drone distances. A calibration layer reduced real-dataset MAE to about **3.03 m**, but this requires labeled real calibration data. Therefore, AirDepth should be understood as a monocular distance-estimation framework whose strongest in-domain model is the RF/XGBoost ensemble, while real-world deployment requires either calibration or stronger domain-generalization training.

---

### B. Drone Detection Module

The detection module receives an RGB image:

$$I_t \in \mathbb{R}^{H \times W \times 3}$$

and returns a bounding box around the target drone:

$$b_t = (u_t, v_t, w_t, h_t),$$

where $(u_t, v_t)$ is the bounding-box center in image coordinates, and $w_t, h_t$ are the box width and height.

In implementation, the bounding box can come from a YOLO-style drone detector or from ground-truth annotations during controlled experiments. The distance-estimation module does not solve detection itself; it assumes that a drone region has already been localized.

The bounding box is important because it affects two parts of the system:

* It defines the image region from which drone-relative depth is extracted.
* It provides geometric scale cues, such as width, height, area, aspect ratio, and image position.

The geometry features used by the model include:

$$w_t, h_t, \frac{w_t}{W}, \frac{h_t}{H}, \frac{w_t h_t}{WH}, \frac{w_t}{h_t}, \frac{u_t}{W}, \frac{v_t}{H}.$$

These features are powerful because object size in the image is strongly related to distance. However, they are also risky: different drone models may have different physical sizes, so the same bounding-box width does not always imply the same true distance. This was one of the main findings from the real-drone evaluation.

To make the model more realistic, later experiments used noisy bounding-box augmentation. Instead of training only on perfect boxes, the system generated scaled versions of each bounding box, such as scale-down and scale-up variants. This simulates detector uncertainty and forces the model to tolerate imperfect detections.

---

### C. Depth Estimation Module

The depth module uses a pretrained monocular depth estimation model, mainly Depth Anything, to produce a relative depth map:

$$R_t = D(I_t),$$

where:

$$R_t \in \mathbb{R}^{H \times W}.$$

The values in $R_t$ are relative depth values, not metric distances. Therefore, AirDepth does not directly interpret $R_t$ as meters. Instead, it extracts relative-depth statistics from the drone region and lets a supervised regression model learn the mapping to metric distance.

For a drone bounding box $b_t$, the system extracts a drone-relative depth value from the corresponding region in $R_t$. Several extraction strategies were tested during the project:

* Center or midpoint depth
* Mean depth inside the box
* Median depth inside the box
* Depth from expanded crops around the drone
* Learned combinations of several depth features

The strongest final full-model configuration used the raw relative-depth feature:

$$d_t = \text{median}(R_t[b_t]),$$

specifically the inner-region median depth inside the drone bounding box. The median was preferred because it is more robust than the mean when the bounding box contains background pixels, edges, propellers, or depth-map artifacts.

The complete feature vector is:

$$\phi_t = [d_t, g_t, m_t],$$

where $d_t$ is the relative-depth feature, $g_t$ is the bounding-box geometry vector, and $m_t$ contains metadata features such as weather and time of day when available.

The metric distance estimator is trained as a supervised regression model:

$$\hat{z}_t = f_\theta(\phi_t).$$

The final strongest model family was an ensemble of Random Forest and XGBoost:

$$\hat{z}_t = \alpha f_{\text{RF}}(\phi_t) + (1-\alpha) f_{\text{XGB}}(\phi_t),$$

with the selected blend approximately:

$$\alpha = 0.44.$$

Thus:

$$\hat{z}_t = 0.44 f_{\text{RF}}(\phi_t) + 0.56 f_{\text{XGB}}(\phi_t).$$

The training objective minimizes metric distance error over labeled examples:

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} |z_i - f_\theta(\phi_i)|.$$

The main evaluation metric is mean absolute error:

$$MAE = \frac{1}{N} \sum_{i=1}^{N} |z_i - \hat{z}_i|.$$

This design is hybrid. The depth map provides learned monocular visual cues, while the bounding-box geometry provides perspective-scale cues. The regression model learns how to combine both.

---

### D. Localization Module

The primary output of AirDepth is the estimated metric depth:

$$\hat{z}_t.$$

This represents the drone distance along the camera optical axis.

If camera intrinsics are known, the estimated depth can be converted into a camera-relative 3D position. Let the camera intrinsic parameters be:

$$f_x, f_y, c_x, c_y,$$

where $f_x, f_y$ are focal lengths and $c_x, c_y$ are principal point coordinates.

Given the bounding-box center $(u_t, v_t)$, the estimated 3D position relative to the camera is:

$$\hat{x}_t = \frac{(u_t - c_x)\hat{z}_t}{f_x},$$

$$\hat{y}_t = \frac{(v_t - c_y)\hat{z}_t}{f_y},$$

$$\hat{p}_t = (\hat{x}_t, \hat{y}_t, \hat{z}_t).$$

This localization is camera-centric. It does not estimate global GPS position, world coordinates, or full drone pose. It estimates where the target drone is relative to the observing camera.

In practice, the most reliable and most evaluated quantity in this project is $\hat{z}_t$, the metric distance. The 3D localization equations are an optional geometric extension once camera calibration is available.

---

### Final Method Summary

AirDepth estimates drone distance from a monocular image by combining detection, relative-depth extraction, bounding-box geometry, and supervised metric regression. The final selected model is not a pure depth model and not a pure geometry model. It is a learned hybrid system: relative depth helps provide visual depth cues, while bounding-box geometry provides strong perspective information. The RF/XGBoost ensemble gave the best synthetic performance, while the real-drone experiments showed that calibration or domain-generalization is necessary when drone size and dataset appearance differ from the synthetic training domain.
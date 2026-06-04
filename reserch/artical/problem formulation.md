
## V. Problem Formulation

### A. Problem Definition

This work addresses the problem of estimating the metric distance of a target drone from a single monocular RGB image. Given an image captured by an observing camera, the goal is to estimate the depth of the target drone along the camera optical axis, without using stereo vision, RGB-D sensing, LiDAR, radar, GPS communication, or any cooperative signal from the target drone.

Formally, given an RGB image

$$I_t \in \mathbb{R}^{H \times W \times 3}$$

captured at time $t$, and containing a visible target drone, the system estimates the target drone distance

$$\hat{z}_t \in \mathbb{R}^{+}$$

where $\hat{z}_t$ is the predicted metric depth in meters.

The problem is separated into two different tasks:

1. **Drone detection**, whose goal is to locate the drone in the image and produce a bounding box.
2. **Drone distance estimation**, whose goal is to estimate the metric distance of the detected drone from the observing camera.

This paper focuses on the second task: estimating metric depth from monocular visual information. The detector or annotation provides a target bounding box, and the proposed method uses information from the RGB image, the drone bounding box, monocular relative-depth estimation, and image-level/contextual features to predict the drone distance.

The problem is challenging because monocular images do not contain absolute scale directly. A small nearby drone and a larger distant drone may produce similar image measurements. In addition, monocular depth models usually produce relative depth rather than metric depth, meaning their output is useful for ordering or comparing depth inside the image but is not directly expressed in meters. The task is further complicated by changes in drone appearance, bounding-box scale, viewing angle, weather, lighting, and the domain gap between synthetic training images and real drone imagery.

### B. Input and Output

Let $I_t$ denote the RGB image at time $t$, with height $H$, width $W$, and three color channels. Let the detected target drone bounding box be

$$b_t = (u_{1,t}, v_{1,t}, u_{2,t}, v_{2,t}),$$

where $(u_{1,t}, v_{1,t})$ and $(u_{2,t}, v_{2,t})$ are the top-left and bottom-right image coordinates of the bounding box.

The bounding-box center is defined as

$$c_t = (u_t, v_t),$$

where

$$u_t = \frac{u_{1,t} + u_{2,t}}{2}, \quad v_t = \frac{v_{1,t} + v_{2,t}}{2}.$$

A monocular depth estimator produces a relative depth map

$$R_t = D(I_t),$$

where $D(\cdot)$ is a pretrained monocular depth model and $R_t \in \mathbb{R}^{H \times W}$. The values in $R_t$ are relative depth values, not metric distances.

From the image, bounding box, and relative-depth map, the system extracts a feature vector

$$\phi_t = \phi(I_t, b_t, R_t, m_t),$$

where $m_t$ may include optional metadata such as weather and time of day when available.

The feature vector may include:

* relative-depth statistics inside or around the drone bounding box,
* bounding-box width, height, area, aspect ratio, and normalized center location,
* image/context metadata such as weather and time of day,

The final estimator is a regression model

$$f_\theta(\cdot)$$

that predicts the metric drone distance:

$$\hat{z}_t = f_\theta(\phi_t).$$

The primary output of this work is therefore the scalar metric depth $\hat{z}_t$, measured in meters.

If camera intrinsics are available, the estimated depth can also be converted into an approximate camera-centric 3D position. Given intrinsic parameters $(f_x, f_y, c_x, c_y)$, the position can be estimated as

$$\hat{x}_t = \frac{(u_t - c_x)\hat{z}_t}{f_x},$$

$$\hat{y}_t = \frac{(v_t - c_y)\hat{z}_t}{f_y},$$

$$\hat{p}_t = (\hat{x}_t, \hat{y}_t, \hat{z}_t).$$

However, the main evaluated prediction in this project is the metric $Z$-distance.

The supervised learning objective is to learn parameters $\theta$ that minimize the distance-estimation error over a training set

$$\mathcal{D} = \{(I_i, b_i, z_i)\}_{i=1}^{N},$$

where $z_i$ is the ground-truth drone distance in meters. The main optimization objective can be written as

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} |z_i - f_\theta(\phi_i)|.$$

The main evaluation metric is mean absolute error:

$$MAE = \frac{1}{N} \sum_{i=1}^{N} |z_i - \hat{z}_i|.$$

Relative error is also used:

$$RE_i = \frac{|z_i - \hat{z}_i|}{z_i}.$$

### C. Assumptions

This work operates under the following assumptions:

* The observing system uses a single monocular RGB camera.
* No stereo camera, RGB-D camera, LiDAR, radar, or external depth sensor is available.
* The target drone is non-cooperative. It does not transmit GPS, altitude, pose, velocity, or distance information.
* The target drone is visible in the image and can be localized by a detector or by an available annotation.
* The distance-estimation model receives a bounding box around the drone, but the research problem is not identical to drone detection. Detection provides image localization; the proposed method estimates metric depth.
* The monocular depth model provides relative depth only. Its output is not assumed to be metrically calibrated.
* The camera intrinsic parameters are assumed to be known or calibrated only when converting the estimated depth into a full camera-centric 3D point.
* The primary target variable is the drone distance along the optical axis, $z$, measured in meters.
* The drone may appear at different distances, scales, viewing angles, lighting conditions, and weather conditions.
* In synthetic experiments, ground-truth distance and controlled metadata are available for supervised training and evaluation.
* In real-world evaluation, metadata may be missing or unknown, and the real drone appearance may differ from the synthetic training drones.
* The method should ideally generalize across drone types, but bounding-box geometry may introduce domain dependence because different drones can have different physical sizes and image scales at the same distance.

A concise way to describe the full problem is:

Given a monocular RGB image containing a detected drone, estimate the drone’s metric distance from the camera using only visual information, relative monocular depth cues, bounding-box geometry, and available contextual features, while avoiding reliance on external depth sensors or cooperative target information.
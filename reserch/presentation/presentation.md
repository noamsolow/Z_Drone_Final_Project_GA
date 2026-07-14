# AirDepth Defense Presentation Plan

Goal: 15-minute judge presentation, about 16-17 slides total, with strong flow, visual-first slides, and very little text.

Core story:

AirDepth estimates the metric distance of a drone from a single RGB image. A monocular image does not directly contain metric scale, so the project turns scale ambiguity into a supervised calibration problem: detect the drone, extract relative depth and bounding-box geometry, learn metric distance with regression, and calibrate for real-world transfer.

Best one-sentence defense:

> Scale ambiguity cannot be removed from a monocular image by geometry alone; AirDepth learns metric scale from labeled examples using bounding-box geometry, relative depth cues, and lightweight real-domain calibration.

## Presentation Timing

Target: 15 minutes.

Recommended pacing:

- Slides 1-5: about 3.5 minutes total.
- Slides 6-10: about 4.5 minutes total.
- Slides 11-15: about 6 minutes total.
- Slides 16-17: about 1.5 minutes total.

The most important slides are 6, 11, 12, 14, and 15. These explain the method, the experiment journey, the final synthetic result, the real-world failure, and the calibration success.

If the presentation must be exactly 16 slides, merge Background and Problem. If it must be exactly 15 slides, also merge Dataset Overview and Synthetic Dataset. If the judges care about motivation and data quality, keep all 17.

## Slide Plan

| # | Slide | What to display | Minimal slide text | What to say |
|---:|---|---|---|---|
| 1 | Title | Large drone image or strong pipeline hero visual | AirDepth: Monocular Drone Distance Estimation | Introduce the team and the goal: estimating drone distance from one RGB image, without LiDAR, stereo, radar, GPS, or target cooperation. |
| 2 | Background | Drone image plus small comparison strip of sensing methods | More drones need better positioning | Drones are becoming more common in civilian and defense environments, creating a growing need to detect, track, and position them. Current depth/range methods such as LiDAR, radar, stereo, and RGB-D are accurate but add cost, weight, power, and calibration complexity. |
| 3 | The Problem | Single camera image with a small drone marked | Detection is not enough | Explain why distance matters for tracking, avoidance, surveillance, and counter-drone systems. The system sees only a 2D image but needs a 3D distance, especially Z-distance from the camera. |
| 4 | Why It Is Hard | Simple visual comparing small-near vs large-far drone projections | Scale ambiguity | A single image loses metric scale. A small nearby drone and a larger far drone can create similar image measurements. This is the central technical challenge. |

| 6 | AirDepth Overview | Clean pipeline diagram | Detect -> Depth -> Features -> Regress -> Calibrate | Explain the full framework. Detection gives the drone region. Monocular depth gives relative cues. Geometry gives apparent scale. Regression learns meters. Calibration adapts to real data. |
| 7 | Dataset Overview | Two-column visual: synthetic on left, real on right | Synthetic for learning, real for transfer | Explain the data strategy. Synthetic data gives scale and labels for training. Real data tests whether the learned model transfers to actual drone images. |
| 8 | Synthetic Dataset | Grid of Unreal images across distances, weather, and time | 15,064 controlled images | Show the synthetic dataset clearly. Mention 20m-150m, clear sky/light rain, 10AM/8PM, and ground-truth distance from the controlled Unreal setup. |
| 9 | Feature Design | Drone bounding box over image, next to depth crop or feature icons | The box is a sensor | Bounding-box geometry was not just a detector output. Width, height, area, aspect ratio, and image position became powerful distance cues. |
| 10 | Depth Feature Experiments | Example depth maps and local crops | Local depth beats global depth | Explain that relative depth alone is not metric. The project tested midpoint depth, mean depth, median depth, local context, and fusion. Tight drone-centered features worked better than broad scene context. |
| 11 | Model Evolution | MAE progress chart | 47.7m -> 6.0m MAE | Tell the experiment journey. Scale-only failed, depth-only failed, geometry helped a lot, tree models helped more, jitter aggregation improved robustness, and RF-XGBoost ensemble became the best synthetic model. |
| 12 | Final Synthetic Result | Model ranking chart or final table | 6.01m MAE, R2 = 0.949 | Present the final held-out synthetic benchmark. Final ensemble: 6.01m MAE, 81.0% within 10m, stable CV-to-test gap of about 0.04m. |
| 13 | Distance Breakdown | Near/mid/far bar chart or heatmap | Harder when farther | Show that error increases with distance because the drone becomes smaller and less stable. Final ensemble: near 3.37m, mid 6.96m, far 9.18m MAE. |
| 14 | Sim-to-Real Failure | Raw real predicted-vs-true plot or before-calibration error plot | Synthetic model overpredicts real data | Be honest: the synthetic model did not directly generalize to real data. Raw real MAE was about 23.80m. The issue was systematic domain shift, especially bounding-box scale and camera/drone appearance. |
| 15 | Calibration Success | Before/after calibration plot | 23.80m -> 3.28m real MAE | This is the real-world payoff. With 20% real calibration data, per-drone quadratic calibration reached 3.28m MAE and 96.4% within 10m. Full real-data fit reached about 3.22m MAE. |
| 16 | Contributions and Limits | Three-column visual: Method, Experiments, Transfer | What we proved | Contributions: target-centric monocular metric distance, systematic ablation studies, robust feature design, and lightweight sim-to-real calibration. Limits: depends on detection quality, real deployment needs calibration, runtime was not fully benchmarked. |
| 17 | Questions | Strong drone image, clean background | Questions? | End with the core message: AirDepth shows monocular drone distance is possible when relative depth is converted through learned geometry-aware calibration. |

## Recommended Visual Assets

Use existing project visuals where possible.

- `poster/synthetic_model_progress_mae.svg`  
  Best for Slide 11. Shows the model improvement story.

- `poster/final_model_mae_table.svg`  
  Useful for Slide 12 or as backup.

- `attempts/attempt8-comprehensive run/studies/study_all/artifacts/plots/poster_test_mae_worst_to_best.svg`  
  Best model ranking chart for Slide 12.

- `attempts/attempt8-comprehensive run/studies/study_all/artifacts/plots/poster_distance_band_mae_heatmap.svg`  
  Good for Slide 13.

- `attempts/attempt8-comprehensive run/studies/study_all/artifacts/plots/poster_best_model_mae_by_true_distance_bars.svg`  
  Alternative for Slide 13.

- `attempts/attempt8-comprehensive run/studies/study02/artifacts/plots/poster_before_after_per_drone_calibration_mae.png`  
  Best for Slide 15.

- `attempts/attempt8-comprehensive run/studies/study02/artifacts/plots/raw_vs_calibrated_predicted_vs_true.png`  
  Good for Slides 14-15.

- `attempts/attempt8-comprehensive run/studies/study02/artifacts/plots/absolute_error_cdf_before_after.png`  
  Good backup visual for calibration impact.

- `experiments/background_drone_exp/`  
  Use for example drone images and depth maps.

- `outputs/poster_review_media/`  
  Review for existing poster-ready images.

## Speaker Notes By Slide

### Slide 1 - Title

Say:

"Our project is AirDepth. The goal is to estimate the distance of a drone from a single RGB image. We focus on the Z-distance from the observing camera, and optionally convert it into camera-relative 3D position when camera intrinsics are available."

Avoid:

- Do not start with implementation details.
- Do not mention every model name yet.

### Slide 2 - Background

Say:

"Drones are becoming more common in both civilian and defense environments. That creates a growing need not only to detect drones, but to track and position them in space. Current range-sensing methods like LiDAR, radar, stereo vision, and RGB-D cameras can help, but they add hardware cost, weight, power consumption, and calibration complexity."

Minimal slide text:

- `More drones`
- `Need tracking + positioning`
- `Current methods: LiDAR / radar / stereo / RGB-D`
- `Problem: cost + weight + calibration`
- `Our direction: RGB only`

Visual idea:

Use a drone image on the left. On the right, show a short comparison strip: heavy sensors on top, RGB-only path at the bottom, with an orange arrow pointing toward AirDepth.

Key point:

This slide explains why the project matters before introducing the technical problem.

### Slide 3 - The Problem

Say:

"For many drone systems, detecting that a drone exists is not enough. We also need to know how far away it is. That distance is important for tracking, collision avoidance, surveillance, and counter-UAV response. Our main target is the Z-distance from the observing camera."

Key point:

The project is about metric distance, not only drone detection.

### Slide 4 - Why It Is Hard

Say:

"The hard part is scale ambiguity. When a 3D world is projected into a 2D image, metric depth is lost. A small nearby drone and a larger distant drone can produce similar measurements in the image. That means a single image cannot give absolute scale by geometry alone."

Strong judge-ready line:

"We do not pretend that monocular scale ambiguity disappears. We handle it by learning and calibrating the mapping from image cues to meters."

### Slide 5 - Existing Solutions Gap

Say:

"LiDAR, radar, stereo vision, and RGB-D cameras can provide depth, but they add hardware cost, payload, power consumption, and calibration complexity. Our question was whether a lighter RGB-only approach can become accurate enough to be useful."

### Slide 6 - AirDepth Overview

Say:

"AirDepth is a split perception pipeline. The detector finds the drone region. A monocular depth model produces a relative depth map. Feature extraction summarizes the drone box, local depth, and metadata. A regression model predicts metric distance. Finally, calibration adapts the prediction to real-world data."

Formula to mention only if needed:

`z_hat = c(f_theta(phi(image, bbox, depth_map, metadata)))`

Keep it visual. Do not put the full formula in large text unless the judges like math.

### Slide 7 - Dataset Overview

Say:

"We used two kinds of data for two different purposes. The synthetic dataset is where we can train and evaluate with controlled ground-truth distances. The real Nenrus data is where we test the transfer problem and calibrate the model to real images."

Important:

Synthetic data is for training and large controlled experiments. Real data is for domain transfer and calibration evaluation.

### Slide 8 - Synthetic Dataset

Say:

"The synthetic benchmark contains 15,064 Unreal Engine images. The distances range from 20m to 150m, with different weather conditions and times of day. This matters because we needed many labeled examples where the real distance is known exactly."

What to show:

- A grid of synthetic drone images.
- A simple label strip with distance, weather, and time of day.
- Optional filename example, but keep it visual.

Good minimal text:

`20-150m | clear/rain | day/evening`

Important:

Do not spend too long here. The purpose of this slide is to convince judges that the dataset is controlled and varied.

### Slide 9 - Feature Design

Say:

"One of the biggest findings is that the bounding box is not only a localization tool. It is a measurement. Width, height, area, aspect ratio, and normalized position carry strong distance information because apparent size changes with distance."

Key point:

Geometry became the strongest synthetic cue, but also a source of real-domain shift.

### Slide 10 - Depth Feature Experiments

Say:

"Monocular depth models give relative depth. We tested many ways to extract useful target-centric cues: center depth, mean box depth, median box depth, local context, object-vs-background contrast, and multi-scale fusion. The best signals were local and object-centered."

Avoid:

Do not overexplain every attempted feature. The point is that you systematically tested options.

### Slide 11 - Model Evolution

Say:

"This slide shows the research path. The simplest scale-only approach was weak. Depth-only was also weak. Adding bounding-box geometry created the first major jump. Nonlinear tree models improved strongly. Jitter aggregation made the features robust to detector noise. The final RF-XGBoost ensemble gave the best synthetic benchmark result."

Numbers to mention:

- Scale-only: about 47.70m MAE in Attempt 8.
- Depth-only linear: about 32.29m MAE.
- Linear + bbox: about 13.20m MAE.
- Random Forest: about 7.19m MAE.
- RF + jitter: about 6.39m MAE.
- RF-XGBoost ensemble: about 6.01m MAE.

### Slide 12 - Final Synthetic Result

Say:

"On the held-out synthetic test split of 2,260 rows, the final ensemble achieved 6.01m MAE, R2 of 0.949, and 81% of predictions within 10m. The cross-validation result was almost identical to the test result, which suggests stable generalization inside the synthetic domain."

### Slide 13 - Distance Breakdown

Say:

"Performance depends on distance. Near targets are easier because the drone occupies more pixels. Far targets are harder because the bounding box and depth features become less stable. The final model still degrades much more gracefully than the linear and depth-only baselines."

Numbers:

- Near: 3.37m MAE.
- Mid: 6.96m MAE.
- Far: 9.18m MAE.

### Slide 14 - Sim-to-Real Failure

Say:

"When we moved from synthetic data to Nenrus real drone images, the raw model failed in a systematic way. It overpredicted almost every real image. The raw real MAE was about 23.80m. This was not random noise; it was domain shift."

Explain domain shift:

- Different camera intrinsics.
- Different drone appearance.
- Different annotation behavior.
- Different bounding-box scale behavior.
- Synthetic rendering does not perfectly match real image statistics.

### Slide 15 - Calibration Success

Say:

"Because the real error was systematic, we tested lightweight calibration instead of retraining the whole model. With only 20% of the real data used for calibration, a per-drone quadratic mapping reduced real MAE from about 23.80m to 3.28m, with 96.4% of predictions within 10m."

Strong line:

"This shows that the model learned useful visual structure, but needed real-domain scale correction."

### Slide 16 - Contributions and Limits

Say:

"The project contributes a full target-centric monocular distance framework, a controlled synthetic benchmark, a systematic ablation path, robust jitter-based feature aggregation, and a practical sim-to-real calibration method."

Limits to state confidently:

- The method depends on good drone detection or annotations.
- Real deployment needs a small calibration set for each camera/drone domain.
- Runtime was not fully benchmarked.
- Far distance remains the hardest regime.

Do not apologize too much. Judges usually trust a project more when limits are stated clearly.

### Slide 17 - Questions

Say:

"To summarize: monocular depth alone is not enough for metric drone distance. AirDepth combines relative depth, bounding-box geometry, supervised regression, and calibration to recover useful metric scale from a single RGB image."

Then stop. Let the judges ask.

## Key Numbers To Memorize

- Synthetic dataset: 15,064 images.
- Synthetic split: 12,804 dev, 2,260 held-out test.
- Synthetic distance range: 20m to 150m.
- Real dataset: 489 Nenrus images.
- Final synthetic model: RF-XGBoost ensemble.
- Final synthetic MAE: 6.01m.
- Final synthetic R2: 0.949.
- Final synthetic within 10m: 81.0%.
- Near/mid/far MAE: 3.37m / 6.96m / 9.18m.
- Raw real MAE: 23.80m.
- Real calibrated MAE with 20% calibration: 3.28m.
- Real calibrated within 10m: 96.4%.

## Likely Judge Questions

### Why not just use the depth model output directly?

Answer:

"Because monocular depth models usually produce relative depth, not metric distance. In our experiments, depth-only regression was much weaker than the final model. The depth map is useful as a cue, but it needs geometry and supervised calibration to become metric."

### How do you solve scale ambiguity?

Answer:

"We do not solve it from the image alone. We turn scale ambiguity into a supervised calibration problem. The model learns from examples how bounding-box geometry, relative depth, and metadata map to real distances."

### Why is bounding-box geometry so important?

Answer:

"Because apparent object size is strongly related to distance under perspective projection. Width, height, and area are strong scale cues. The risk is that geometry can be domain-dependent, which is why real-domain calibration is important."

### Why did the synthetic model fail on real images?

Answer:

"The failure was mostly systematic domain shift. Real images have different camera properties, drone scale behavior, annotation patterns, sharpness, and appearance. The model overpredicted real distances, and calibration corrected that scale bias."

### Is calibration cheating?

Answer:

"No. Calibration is a standard deployment step when transferring a visual model to a new camera or target domain. Importantly, we did not retrain the full RF-XGBoost model. We used a lightweight mapping from raw prediction to real-domain distance."

### What is the main innovation?

Answer:

"The innovation is the target-centric hybrid design: using monocular relative depth as a cue, not as the answer, and combining it with bounding-box geometry, robust jitter aggregation, and real-domain calibration for metric drone distance."

### What would you do next?

Answer:

"First, integrate a full real detector into the runtime pipeline. Second, benchmark runtime. Third, collect more real calibration data across more drone types and cameras. Fourth, add temporal smoothing for video, such as a Kalman filter."

## Final Story Arc

Use this structure mentally:

1. Metric drone distance from one RGB image is useful but hard.
2. It is hard because monocular scale is ambiguous.
3. Relative depth alone is not enough.
4. Bounding-box geometry is extremely informative.
5. Tree-based regression learns the nonlinear mapping to meters.
6. Jitter aggregation makes the model more robust to detector noise.
7. Synthetic results are strong: 6.01m MAE.
8. Real transfer initially fails because of domain shift.
9. Lightweight calibration fixes most of the real-domain bias.
10. AirDepth is a practical RGB-only framework, with honest limitations.

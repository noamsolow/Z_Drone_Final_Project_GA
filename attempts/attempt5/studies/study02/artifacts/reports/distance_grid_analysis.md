# Distance Grid Effect Analysis

Training distances in the exported model's original Attempt 4 feature table:

`20, 30, 40, 50, 60, 70, 80, 90, 100, 115, 125, 150`

## Question

Does the external Nenrus error get worse mainly because some Nenrus distances are not multiples of 10, or because they are not distances seen during training?

## Multiple-of-10 vs Non-Multiple-of-10

- `False`: count `285`, MAE `25.329m`, mean relative error `1.117`, mean signed error `25.329m`
- `True`: count `204`, MAE `27.196m`, mean relative error `0.838`, mean signed error `27.196m`

## Exact Training Distance vs Unseen Exact Distance

- `False`: count `304`, MAE `24.382m`, mean relative error `1.110`, mean signed error `24.382m`
- `True`: count `185`, MAE `28.943m`, mean relative error `0.819`, mean signed error `28.943m`

## Gap From Nearest Training Distance

- `exact_seen_training_distance`: count `185`, MAE `28.943m`, mean relative error `0.819`, mean signed error `28.943m`
- `more_than_10m_from_training_distance`: count `50`, MAE `15.017m`, mean relative error `2.644`, mean signed error `15.017m`
- `within_10m_of_training_distance`: count `19`, MAE `10.184m`, mean relative error `1.018`, mean signed error `10.184m`
- `within_5m_of_training_distance`: count `235`, MAE `27.523m`, mean relative error `0.792`, mean signed error `27.523m`

## Below Minimum Training Distance

- `False`: count `376`, MAE `29.673m`, mean relative error `0.780`, mean signed error `29.673m`
- `True`: count `113`, MAE `14.244m`, mean relative error `1.734`, mean signed error `14.244m`

## Interpretation

The coarse training grid probably contributes to the problem, but it is not the full explanation.
The strongest evidence is that exact training distances in Nenrus still have high external error.
The very short distances below 20m have especially large relative error because the model was never trained below 20m and all predictions are biased upward.

So the issue has two layers:

- interpolation/grid issue: unseen distances such as 25, 35, 45, 55, 65, 75 are not exact training targets
- extrapolation/domain issue: distances below 20m are outside the training range, and the real-drone geometry differs from the original dataset

The observed failure is therefore not just because some labels are 4m or 9m instead of clean tens. It is also because the model learned a same-domain geometry-to-distance mapping that does not transfer cleanly to Nenrus.

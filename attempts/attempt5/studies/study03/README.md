# Study 04: Depth-Weighted External Variants

This study tests whether making relative depth more influential can improve
external-domain performance on Nenrus.

The hypothesis is:

- geometry is causing the domain shift
- maybe stronger relative-depth dependence can reduce that failure

The study trains only on the original Attempt 4 feature table and evaluates on
Nenrus. It does not train on Nenrus labels.

## Variants

The study compares:

- current saved ensemble baseline
- `depth_only`
- `depth_plus_no_size_geometry`
- `depth_plus_normalized_geometry`
- `depth_repeated_8_with_geometry`
- `depth_repeated_16_with_geometry`

The repeated-depth variants duplicate the same raw relative-depth feature in the
tabular matrix. This gives RF more chances to consider depth during feature
subsampling and tests whether a stronger depth signal helps in practice.

## Run

```powershell
.\.venv\Scripts\python.exe "attempts/attempt5/studies/study03/run_depth_weighted_variants.py"
```

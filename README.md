# CH5440 — MVDA Assignment 2

MATLAB implementation of multivariate regression methods for CH5440 (Multivariate Data Analysis).

## Questions

| Script | Topic |
|--------|-------|
| `scripts/question1.m` | Quadratic polynomial regression (power cell data) |
| `scripts/question2.m` | Mixture regression — Scheffe model (yarn elongation) |
| `scripts/question3.m` | Logistic regression (hiring decisions) |
| `scripts/question4.m` | Logistic regression (chocolate purchases) |
| `scripts/question5.m` | Binomial MLE — coupon redemption (grouped) |
| `scripts/question6.m` | Bernoulli MLE — coupon redemption (ungrouped) |
| `scripts/question7.m` | Multivariate logistic regression (dengue disease) |

## Running

Open MATLAB, navigate to the repo root, and run:

```matlab
>> cd scripts
>> question1
```

Or from terminal:
```bash
matlab -r "cd scripts; question1; exit"
```

Requires the Statistics and Machine Learning Toolbox.

## Structure

```
scripts/    MATLAB source files
data/       Input data files
output/     Console output (.txt) and figures (.png)
report/     LaTeX source and compiled PDF
assignment/ Original problem statement
```

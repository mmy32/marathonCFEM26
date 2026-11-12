# Steps Moving Forward

**Goal:** build a model to predict non-accrual in borrower loans.

**Team:** Olivia, Alex, Mincy, Max, Young

**Unit of observation:** borrower-quarter (loans pooled across BDCs).

**Outcomes**

1. Non-accrual within the next quarter (binary).
2. Markdown in valuation (continuous), measured two ways: `val_ret` and price change.

**Inputs:** mark level, rolling max drawdown, PIK switch, valuation dispersion.

**Baseline:** one simple model per predictor versus each outcome. Logistic regression for non-accrual; simple linear regression for the two markdown outcomes (logistic does not fit a continuous response).

---

## Due next week

- [ ] Agree on the outcome definitions (non-accrual next quarter; `val_ret` and price change) and the borrower-quarter unit
- [ ] Build the four predictors at the borrower-quarter level
- [ ] Fit the baseline models, one predictor at a time, for each outcome
- [ ] Collect the first results (coefficients, fit, AUC for the logits)
- [ ] Draft slides for each presentation topic below

## Presentation

| Topic | Speaker |
|---|---|
| The outcome variables being predicted (non-accrual; markdown as `val_ret` and price change) | Max |
| The choices of models | Alex |
| The input predictor variables | Young |
| Results and further exploration | Olivia and Mincy |

## Model build

### 1. Define the outcome variable
- [ ] Non-accrual within the next quarter: define the 0/1 flag from the non-accrual data
- [ ] Decide how to treat borrowers already on non-accrual at the start of the quarter
- [ ] Markdown outcomes: `val_ret` (next quarter) and price change (FV/PAR)
- [ ] Confirm the borrower-quarter unit and the sample filters (e.g. minimum prior fair value)

### 2. Define the input variables
- [ ] Mark level
- [ ] Rolling max drawdown (choose the window)
- [ ] PIK switch (define the switch: when a borrower starts paying in kind)
- [ ] Valuation dispersion (spread of marks across BDCs holding the borrower)
- [ ] Check that every predictor is measured before the outcome quarter, with no look-ahead

### 3. Select models
- [ ] Baseline: one logistic regression per predictor for non-accrual
- [ ] Baseline: one simple linear regression per predictor for each markdown outcome
- [ ] Combined logistic (non-accrual) and linear (markdown) regression with all four inputs
- [ ] Regularized regression (lasso / ridge)
- [ ] Gradient-boosted trees
- [ ] Neural network (small MLP)

### 4. Build out
- [ ] Assemble the borrower-quarter panel with outcomes and lagged predictors
- [ ] Split train and test by time, not at random
- [ ] Fit the models and save the outputs

### 5. Evaluation
- [ ] Classification: AUC and a confusion table at a stated threshold
- [ ] Regression: R² and out-of-sample error
- [ ] Compare predictors against each other and against the baseline
- [ ] Note the limits (rare events, overlapping quarters, sample size)

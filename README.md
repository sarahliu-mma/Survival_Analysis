# Survival Analysis: Predicting the Probability of a Wildfire Reaching Evacuation Zones Within 12, 24, 48, and 72 Hours

*This project was inspired by the WiDS Global Datathon 2026, a hackathon hosted on Kaggle. More information can be found at: [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview)*

---

## Problem Overview

- Predict the probability of a wildfire reaching within 5 km of an evacuation zone centroid within 12, 24, 48, and 72 hours
- Because the dataset contains censored observations, this is a time-to-event problem best addressed with survival analysis
- The target columns are `time_to_hit_hours` and `event`. The final output is four survival probabilities per wildfire — the likelihood that no hit occurs within 12, 24, 48, and 72 hours from the observation time

---

## Dataset

- **Source:** [WiDS Global Datathon 2026 Data](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/data)
- 221 training rows and 95 test rows
- Event rate of ~31%, making this a small, imbalanced dataset
- Each row contains 34 features plus 2 target columns and 1 identifier column (train only)

```python
# Load the data
path = "/teamspace/studios/this_studio/WiDS/"

train = pd.read_csv(path + "train.csv")
test  = pd.read_csv(path + "test.csv")
meta  = pd.read_csv(path + "metaData.csv")

print(f"Train: {train.shape}")
print(f"Test:  {test.shape}")

# Define target variables
duration_col = 'time_to_hit_hours'
event_col    = 'event'
```

---

## Methodology

### EDA & Understanding the Data

The first step is checking for missing values. We also examine whether numeric features are skewed, since the Cox model assumes the hazard ratio remains constant over time (the proportional hazards assumption). Leaving heavily skewed features untransformed can violate this assumption and make coefficients harder to interpret.

#### Censoring Rate

The censoring rate matters for three reasons. First, it reflects how much signal is available for the model — a high censoring rate can cause underfitting. Second, it informs model selection: a censoring rate in the 30–60% range is a strong signal that survival analysis is appropriate. Third, when censoring is high, relying solely on the C-index may be insufficient, and a permutation test may be warranted.

```python
print(f'Event rate:     {train[event_col].mean():.2%}')
print(f'Censoring rate: {(train[event_col] == 0).mean():.2%}')
print(f'Duration stats:\n{train[duration_col].describe()}')
```

#### Kaplan-Meier Curve

The KM curve shows how survival probability evolves over time. The x-axis represents time (hours) and the y-axis represents the probability the event has not yet occurred. A steep early decline indicates higher early risk. The curve also reveals where censored observations cluster, which helps detect potential informative censoring. The shaded confidence interval widens over time as the number of remaining observations decreases.

```python
kmf = KaplanMeierFitter()
kmf.fit(durations=train[duration_col], event_observed=train[event_col])

plt.figure(figsize=(9, 5))
kmf.plot_survival_function(ci_show=True)
plt.title('Kaplan-Meier Survival Curve — All Zones')
plt.xlabel('Time to Hit (hours)')
plt.ylabel('Probability of Not Being Hit')
plt.axhline(0.5, linestyle='--', alpha=0.6, color='red', label='50% threshold')
plt.legend()
plt.show()
```

The curve shows that at least 80% of wildfires had not reached a zone within the first 10 hours. The widening confidence interval over time reflects increasing uncertainty as censored observations accumulate.

> **Note:** Cyclical encoding for time-based features was not applied here, as the dataset already provides pre-encoded temporal columns.

---

### Feature Engineering

- **Log transforms** applied to skewed numeric features to reduce the influence of outliers and better satisfy the proportional hazards assumption
- No missing value imputation was needed, as EDA confirmed no missing values in the dataset

---

### Preprocessing

- **Feature scaling:** The Cox model is sensitive to feature scale, so all features were standardized (mean = 0, std = 1). The test set was scaled using the parameters fitted on the training set — not refit — to prevent data leakage.
- **Train/validation split:** A holdout validation set was used to evaluate model performance before generating test predictions.

---

### Model

#### Cox Proportional Hazards (Baseline)

- **Library:** `lifelines`
- The Cox model requires all data (features + targets) in a single DataFrame, so separate DataFrames were prepared for train, validation, and test sets before fitting.
- **Penalizer tuning:** Multiple Cox models were fit with different penalizer values. The penalizer that maximized validation C-index was selected. At a penalizer of **0.5**, the model achieved the highest validation C-index of **0.8742**.
- **Assumption checks:** Schoenfeld residuals and log-log plots were used to assess the proportional hazards assumption. Results looked reasonable given the small sample size.

#### Coefficient Interpretation

- `dist_min_ci_0_5h` had a small p-value, indicating a statistically significant coefficient
- `cross_track_component` had a confidence interval crossing zero, suggesting its coefficient may not be reliably estimated

#### Validation Performance

| Metric | Value |
|---|---|
| Validation C-index | 0.8742 |

---

### Final Submission

Survival functions were predicted for all test zones and converted into the four required probability columns (`prob_12h`, `prob_24h`, `prob_48h`, `prob_72h`).

---

## Limitations

- **Small sample size:** With only 221 training rows and ~51 events, the model has limited statistical power. Coefficient estimates are unstable, and assumption tests (e.g., Schoenfeld residuals) should be interpreted cautiously.
- **Proportional hazards assumption:** The Cox model requires hazard ratios to remain constant over time. While checks looked reasonable, this assumption is difficult to validate reliably on a dataset this small.
- **Single model:** The current pipeline relies solely on Cox PH, which is a linear model. It may not capture non-linear relationships or complex feature interactions present in the data.

---

## Future Work

- **Random Survival Forest (RSF):** RSF makes no proportional hazards assumption and naturally handles non-linear relationships and feature interactions. It is a strong candidate to replace or complement the Cox baseline.
- **XGBoost/LightGBM with Cox objective:** Gradient boosted models with a Cox loss function can capture complex patterns while remaining computationally efficient on small datasets.
- **Ensembling:** Averaging risk scores from Cox PH, RSF, and a boosted survival model could reduce variance and improve generalization.

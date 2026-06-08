# Survival Analysis: Prediction the probability of a wildfire reaching evacuation zones within 12,24,48, and 72 hours
This project was inspired by the WiDS Global Datathon 2026, a hackathon hosted on Kaggle. More information can be found at: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview 

Since the complete details are already available on the Kaggle website, this document focuses on explaining the rationale, goals, and code logic for each step. 

## Project Goal 

Predict the probability of a wildfire reaching within 5 km of the evacuation zones of centroid within 12, 24, 48, and 72 hours. 

## Process

Before we begin our analysis, let's make sure we've already installed 'lifelong' module for conducting survival analysis.
Also, don't forget to import the tools we need.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
```

### Step 1: EDA & Understanding the Data 
  The major goal of this section is to address the following questions: 
  1. What is the censoring rate? 
  2. Which features are associated with the event (i.e., the wildfire reaching the area within 72 hours)? 
  3. Are there any missing values or outliers? 
  4. Do any feature distributions require transformation? (Since the log-transformed data is already available in the dataset, this step is skipped.) 
```python
# Load the data
path = "/teamspace/studios/this_studio/WiDS/"

train = pd.read_csv(path + "train.csv")
test  = pd.read_csv(path + "test.csv")
meta  = pd.read_csv(path + "metaData.csv")

print(f"Train: {train.shape}")
print(f"Test:  {test.shape}")
train.head()
test.head()
```
```python
# Define the target variables
duration_col = 'time_to_hit_hours'
event_col = 'event'
```

### Censoring Rate

Why does the censoring rate matter? First, it indicates how difficult it is for a model to learn from the available data — a high censoring rate may cause underfitting. Second, it guides model selection: if censored observations account for roughly 30–60% of the data, a survival analysis model is appropriate. Finally, when the censoring rate is high, a permutation test is necessary, as the C-index alone may not provide sufficient evidence of model performance.
```python
#Take a peak at the target variables
print(f'Event rate: {train[event_col].mean():.2%}')
print(f'Cencoring rate: {(train[event_col] == 0).mean():.2%}')
print(f'Duration Stats: {train[duration_col].describe()}')
```
```python
#Check missing values
missing = train.isnull().sum()
missing = missing[missing>0].sort_values(ascending=False)
print("Missing values:")
print(missing)
```
```python
#Check skewness of numeric columns
num_cols = train.select_dtypes(include = 'number').columns.tolist()
skewness = train[num_cols].skew().sort_values(ascending=False)
print('Top skewed columns:')
print(skewness.head(10))
```

### Kaplan-Meier Curve

The KM curve illustrates how risk evolves over time. The x-axis represents time (in hours), and the y-axis represents the probability that the event has not yet occurred. A steeper early decline indicates a higher risk of the wildfire reaching the area sooner. The KM curve also reveals the distribution of censored observations, allowing one to detect whether censored data clusters in specific time ranges. Additionally, it provides an overview of the overall data distribution. 

```python
#See overall survival pattern before any modeling
kmf = KaplanMeierFitter()
kmf.fit(durations = train[duration_col], event_observed=train[event_col])

plt.figure(figsize = (9,5))
kmf.plot_survival_function(ci_show = True)
plt.title('Kaplan-Meier survival curve — all zones')
plt.xlabel('Time to hit (hours)')
plt.ylabel('Probability of not being hit')
plt.axhline(0.5, linestyle='--', alpha=0.6, color='red', label='50% threshold')
plt.legend()
plt.show()
```

# Survival Analysis: Prediction the probability of a wildfire reaching evacuation zones within 12,24,48, and 72 hours
This project was inspired by the WiDS Global Datathon 2026, a hackathon hosted on Kaggle. More information can be found at: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview 

Since the complete details are already available on the Kaggle website, this document focuses on explaining the rationale, goals, and code logic for each step. 

## Project Goal 

Predict the probability of a wildfire reaching within 5 km of the evacuation zones of centroid within 12, 24, 48, and 72 hours. 

## Process

Before we begin our analysis, let's make sure we've already installed 'lifelong' module for conducting survival analysis

### Step 1: EDA & Understanding the Data 
  The major goal of this section is to address the following questions: 
  1. What is the censoring rate? 
  2. Which features are associated with the event (i.e., the wildfire reaching the area within 72 hours)? 
  3. Are there any missing values or outliers? 
  4. Do any feature distributions require transformation? (Since the log-transformed data is already available in the dataset, this step is skipped.) 

### Censoring Rate

Why does the censoring rate matter? First, it indicates how difficult it is for a model to learn from the available data — a high censoring rate may cause underfitting. Second, it guides model selection: if censored observations account for roughly 30–60% of the data, a survival analysis model is appropriate. Finally, when the censoring rate is high, a permutation test is necessary, as the C-index alone may not provide sufficient evidence of model performance. 

### Kaplan-Meier Curve

The KM curve illustrates how risk evolves over time. The x-axis represents time (in hours), and the y-axis represents the probability that the event has not yet occurred. A steeper early decline indicates a higher risk of the wildfire reaching the area sooner. The KM curve also reveals the distribution of censored observations, allowing one to detect whether censored data clusters in specific time ranges. Additionally, it provides an overview of the overall data distribution. 

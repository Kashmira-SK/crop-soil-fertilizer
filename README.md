

# Precision Fertilizer Recommendation

A machine learning project exploring fertilizer classification using soil properties, nutrient levels, crop information, and environmental conditions.

The project implements and compares seven classification algorithms, covering data preprocessing, exploratory analysis, feature engineering, hyperparameter tuning, and model evaluation.

An important part of the project was investigating why the original dataset produced consistently poor predictions despite being clean and balanced. A second dataset was introduced to compare model behavior and determine whether the limitations came from the data or the modeling approach.

## Overview

The objective was to develop a classification pipeline capable of predicting fertilizer types from agricultural measurements.

The project covers:

- Data cleaning and exploratory data analysis (EDA)
- Outlier detection and treatment
- Feature engineering and selection
- Categorical encoding and feature scaling
- Training and tuning seven classification models
- Stratified cross-validation and performance evaluation
- Comparative analysis across two fertilizer datasets

Rather than relying on a single algorithm, multiple model families were evaluated to understand how different approaches performed on the same data.

## Tech Stack

**Language:** Python

**Libraries:**
- pandas and NumPy — Data manipulation and numerical operations
- scikit-learn — Preprocessing, classification, hyperparameter tuning, and evaluation
- Matplotlib — Data visualization and exploratory analysis

**Environment:**
- Jupyter Notebook

## Datasets

Two datasets were used to evaluate the classification pipeline.

### 1. Crop and Soil Dataset

[Dataset on Kaggle](https://www.kaggle.com/datasets/shankarpriya2913/crop-and-soil-dataset)

The original dataset contains 8,000 records with eight input features and seven fertilizer classes.

**Features:**
- Temperature
- Humidity
- Moisture
- Nitrogen
- Potassium
- Phosphorous
- Soil Type
- Crop Type

**Target:** `Fertilizer Name`

Initial inspection found no missing values or duplicate records, and the fertilizer classes were approximately balanced.

Despite these characteristics, the dataset produced consistently weak classification results.

### 2. Western Maharashtra Fertilizer Dataset

[Dataset on Kaggle](https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra)

An additional dataset containing 4,513 records, nine input features, and 19 fertilizer classes.

It includes soil chemistry, rainfall, temperature, crop type, soil colour, and geographic information.

**Target:** `Fertilizer`

This dataset was used to investigate the limitations observed in the original data and evaluate whether the same models could learn more effectively from different feature–target relationships.

Unlike the original dataset, its target distribution is significantly imbalanced, making class-sensitive evaluation particularly important.

## Repository Structure

- `data/raw/` — Original assigned dataset
- `data/external/` — External comparison datasets
- `notebooks/` — Individual preprocessing notebooks
- `group_pipeline.ipynb` — Combined preprocessing pipeline
- `results/eda_visualizations/` — Exploratory data analysis plots
- `results/logs/` — Execution logs, where applicable
- `results/outputs/` — Processed datasets and generated outputs

## Data Processing Pipeline

The preprocessing workflow was designed to prepare both datasets consistently while preventing data leakage.

### 1. Data Validation and Exploration

Inspected dataset structure, data types, missing values, duplicates, feature distributions, and correlations.

Exploratory analysis was used to understand the input variables and identify potential preprocessing requirements.

### 2. Train-Test Splitting

Applied an 80/20 stratified train-test split using `random_state=42`.

Data-dependent transformations were fitted using the training data only, with the resulting transformations applied to the held-out test data.

### 3. Outlier Treatment

Used the Interquartile Range (IQR) method to identify extreme values.

Outliers were capped at the calculated bounds rather than removing entire observations, preserving the dataset size.

### 4. Feature Engineering

Created additional nutrient-based features:

- `Total_Nutrients` — Sum of nitrogen, phosphorus, and potassium
- `NPK_Imbalance` — Difference between the highest and lowest nutrient values

Correlation analysis was used to identify redundant features. `Total_Nutrients` was removed where highly redundant, while `NPK_Imbalance` was retained.

### 5. Encoding and Scaling

- One-hot encoding for categorical inputs
- StandardScaler for numerical inputs
- LabelEncoder for fertilizer classes

Unknown categorical values were handled during transformation to support previously unseen categories.

## Machine Learning Models

Seven classification algorithms were implemented and evaluated.

| Algorithm | Approach |
| --- | --- |
| Decision Tree | Rule-based classification |
| Random Forest | Bagging ensemble |
| Logistic Regression | Linear classification |
| K-Nearest Neighbors | Distance-based classification |
| Support Vector Machine | Kernel-based classification |
| Multilayer Perceptron | Neural network |
| Gradient Boosting | Sequential boosting ensemble |

Each model was tuned using `GridSearchCV` with five-fold stratified cross-validation.

The tuning process used **macro F1** as its scoring metric to account for class-level performance, particularly on the imbalanced comparison dataset.

## Model Performance

### Original Dataset

The original dataset produced consistently poor results across all seven models, even after hyperparameter tuning.

| Model | Test Accuracy | Macro F1 |
| --- | ---: | ---: |
| Decision Tree | 14.50% | 14.42% |
| Random Forest | 14.81% | 14.77% |
| Logistic Regression | 12.75% | 12.56% |
| KNN | 13.81% | 13.80% |
| SVM | 15.50% | 15.41% |
| MLP | 13.75% | 13.46% |
| Gradient Boosting | 13.31% | 13.16% |

The majority-class baseline was approximately **14.88%**, meaning that even the strongest model offered little improvement over predicting the most common class.

The similarity in performance across linear models, neural networks, distance-based methods, and tree ensembles suggested that the issue was not limited to a particular algorithm.

### Comparison Dataset

Applying the same modeling approach to the Western Maharashtra dataset produced substantially stronger results.

| Model | Test Accuracy | Test Macro F1 | CV Macro F1 |
| --- | ---: | ---: | ---: |
| Decision Tree | **93.24%** | 83.68% | 83.36% |
| Random Forest | 82.95% | 79.88% | 77.37% |
| Logistic Regression | 41.09% | 36.63% | 43.86% |
| KNN | 86.71% | 83.76% | 81.95% |
| SVM | 51.50% | 59.98% | 58.05% |
| **MLP** | 92.58% | **92.54%** | **84.13%** |
| Gradient Boosting | 61.24% | 69.88% | 66.77% |

### Selected Model: Multilayer Perceptron

The tuned MLP achieved the highest mean cross-validation macro F1 score on the comparison dataset.

**Best configuration:**
- Hidden layers: `(100, 50)`
- Alpha: `0.001`
- Initial learning rate: `0.01`

**Performance:**
- Test Accuracy: **92.58%**
- Test Macro F1: **92.54%**
- Test Weighted F1: **92.55%**
- Cross-Validation Macro F1: **84.13%**

Although Decision Tree achieved a higher test accuracy of 93.24%, MLP was selected based on the predefined cross-validation macro F1 criterion rather than selecting a model using the held-out test results.

## Key Findings

### Dataset Quality Matters Beyond Cleanliness

The original dataset contained no missing values, no duplicate records, and relatively balanced classes.

However, none of the seven classification algorithms could reliably distinguish its fertilizer labels.

This demonstrated that a dataset can be structurally clean while still lacking the predictive relationships needed for effective classification.

### Model Complexity Does Not Guarantee Better Predictions

More complex algorithms and additional hyperparameter tuning did not meaningfully improve performance on the original dataset.

The consistently weak results suggested a limitation in the available feature–target relationships rather than a problem that could be resolved simply by changing algorithms.

### Evaluation Metrics Influence Model Selection

The comparison dataset contains significant class imbalance.

Using macro F1 alongside accuracy helped account for performance across fertilizer classes, including those with very few samples.

This also influenced the final model selection: MLP was selected through cross-validation even though Decision Tree achieved the highest held-out accuracy.

## Limitations and Future Improvements

While the comparison dataset demonstrated strong predictive performance, the models have not been validated for real-world agricultural deployment.

Current limitations include:

- Incomplete documentation of measurement units and data collection methods
- Geographic limitations of the comparison dataset
- Very small sample sizes for some fertilizer classes
- Lack of independent field validation

Potential improvements include expanding the datasets with verified agricultural measurements, evaluating generalization across different regions, investigating feature importance, and improving minority-class coverage.

## Conclusion

This project demonstrates an end-to-end machine learning workflow, from initial data analysis and preprocessing through model tuning, evaluation, and comparison.

The most significant result was identifying the limitations of the original dataset through repeated experiments across different model families.

By testing the same approach on a second dataset, the project demonstrated that the workflow could produce strong classification results when more useful predictive relationships were present.

The final comparison selected an MLP classifier with **92.58% test accuracy and 92.54% macro F1**, while also highlighting the importance of dataset quality, appropriate evaluation metrics, and evidence-based model selection.

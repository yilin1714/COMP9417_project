# Air Quality Forecasting Project

## Project Overview

This is a machine learning-based air quality forecasting project that uses the UCI Air Quality dataset for pollutant concentration prediction and classification. The project implements both regression and classification tasks, supporting hourly and daily time granularities.

### Key Features

- **Regression Task**: Predict concentrations of multiple pollutants (CO(GT), NOx(GT), NO2(GT), C6H6(GT))
- **Classification Task**: Predict future pollution levels (low/mid/high) at 1/6/12/24 hours ahead
- **Multiple Time Granularities**: Supports both hourly and daily predictions
- **End-to-End Pipeline**: Complete workflow from data preprocessing, feature engineering, model training, evaluation, to visualization

---

## Project Structure

```
project/
├── config/                          # Configuration files
│   ├── global.yaml                  # Global configuration (data paths, model parameters, etc.)
│   ├── lstm.yaml                    # LSTM model configuration
│   └── random_forest.yaml           # Random Forest configuration
│
├── data/                            # Data directory
│   ├── raw/                         # Raw data
│   │   └── AirQualityUCI.csv        # UCI Air Quality raw dataset
│   ├── processed/                   # Preprocessed data
│   │   └── air_quality_clean.csv    # Cleaned data
│   ├── features/                    # Feature-engineered data
│   │   ├── hourly_features.csv      # Hourly features
│   │   ├── daily_features.csv       # Daily features
│   │   └── classification_features.csv  # Classification task features
│   └── splits/                      # Dataset splits (train/val/test)
│       ├── X_train_hourly.csv
│       ├── y_train_hourly.csv
│       ├── X_train_daily.csv
│       ├── y_train_daily.csv
│       ├── X_train_classification.csv
│       ├── y_train_classification_t+1.csv  # Predict t+1
│       ├── y_train_classification_t+6.csv  # Predict t+6
│       └── ...
│
├── src/                             # Source code
│   ├── data/                        # Data processing module
│   │   ├── preprocess.py            # Data preprocessing (cleaning, missing value handling, etc.)
│   │   ├── feature_engineering_hourly.py    # Hourly feature engineering
│   │   ├── feature_engineering_daily.py     # Daily feature engineering
│   │   ├── feature_engineering_classification.py  # Classification task feature engineering
│   │   ├── split_dataset.py         # Regression task dataset splitting
│   │   └── split_dataset_classification.py  # Classification task dataset splitting
│   │
│   ├── models/                      # Model module
│   │   ├── regression_models.py     # Regression model definitions
│   │   ├── classification_models.py # Classification model definitions
│   │   ├── train_regression.py      # Regression model training
│   │   ├── train_classification.py  # Classification model training
│   │   ├── evaluate_regression.py   # Regression model evaluation
│   │   └── evaluate_classification.py       # Classification model evaluation
│   │
│   ├── visualization/               # Visualization module
│   │   ├── plot_pollutants.py       # Pollutant time series visualization
│   │   ├── plot_correlation_analysis.py     # Correlation analysis
│   │   ├── plot_meteorological_relations.py # Meteorological-pollutant relationships
│   │   ├── plot_time_patterns.py    # Time pattern analysis
│   │   └── plot_classification_summary.py   # Classification results summary visualization
│   │
│   └── analysis/                    # Analysis module
│       └── detect_anomalies.py      # Anomaly detection
│
└── results/                         # Results directory
    ├── regression/                  # Regression task results
    │   ├── hourly/                  # Hourly regression results
    │   │   ├── best_models/         # Best model files (.joblib)
    │   │   ├── plots/               # Visualization plots
    │   │   ├── regression_val_metrics.csv
    │   │   └── test_metrics.csv
    │   └── daily/                   # Daily regression results
    │       └── ...
    ├── classification/              # Classification task results
    │   ├── t+1/                     # t+1 prediction results
    │   │   ├── LogisticRegression_best.joblib
    │   │   ├── RandomForest_best.joblib
    │   │   ├── MLP_best.joblib
    │   │   ├── plots/               # Confusion matrices, time series, etc.
    │   │   └── metrics_t+1.csv
    │   ├── t+6/                     # t+6 prediction results
    │   ├── t+12/                    # t+12 prediction results
    │   ├── t+24/                    # t+24 prediction results
    │   ├── classification_val_summary.csv
    │   └── classification_test_summary.csv
    └── plots/                       # General visualization results
        ├── pollutant_timeseries.png
        ├── pollutant_correlation_heatmap.png
        ├── meteo_pollutant_correlation_heatmap.png
        └── timepattern_*.png
```

---

## Quick Start

### 1. Requirements

- Python 3.7+
- Main dependencies:
  - `pandas` - Data processing
  - `numpy` - Numerical computation
  - `scikit-learn` - Machine learning models
  - `matplotlib` / `seaborn` - Visualization
  - `pyyaml` - Configuration file reading
  - `joblib` - Model saving/loading

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml joblib
```

### 3. Data Preparation

Ensure the raw data file `AirQualityUCI.csv` is placed in the `data/raw/` directory.

### 4. Running the Pipeline

#### Step 1: Data Preprocessing

Clean raw data, handle missing values and outliers:

```bash
cd src/data
python preprocess.py
```

Output: `data/processed/air_quality_clean.csv`

#### Step 2: Feature Engineering

Generate features according to task requirements:

**Regression Task - Hourly Features:**
```bash
python feature_engineering_hourly.py
```

**Regression Task - Daily Features:**
```bash
python feature_engineering_daily.py
```

**Classification Task Features:**
```bash
python feature_engineering_classification.py
```

#### Step 3: Dataset Splitting

**Regression Task:**
```bash
python split_dataset.py  # Automatically called during feature processing
```

**Classification Task:**
```bash
python split_dataset_classification.py
```

#### Step 4: Model Training

**Regression Model Training:**
```bash
cd src/models
python train_regression.py
```

Trained models include:
- Linear Regression
- Random Forest Regressor
- MLP Regressor

**Classification Model Training:**
```bash
python train_classification.py
```

Trained models include:
- Logistic Regression
- Random Forest Classifier
- MLP Classifier

#### Step 5: Model Evaluation

**Regression Model Evaluation:**
```bash
python evaluate_regression.py
```

**Classification Model Evaluation:**
```bash
python evaluate_classification.py
```

#### Step 6: Visualization Analysis

**Pollutant Time Series:**
```bash
cd src/visualization
python plot_pollutants.py
```

**Correlation Analysis:**
```bash
python plot_correlation_analysis.py
```

**Meteorological-Pollutant Relationships:**
```bash
python plot_meteorological_relations.py
```

**Time Pattern Analysis:**
```bash
python plot_time_patterns.py
```

**Classification Results Summary:**
```bash
python plot_classification_summary.py
```

---

## Configuration

Main configuration file: `config/global.yaml`

### Key Configuration Items

```yaml
# Data configuration
data:
  targets: ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"]  # Target pollutants
  missing_value: -200        # Missing value marker
  split_method: "year"       # Split method: by year or ratio
  granularity: "daily"       # Time granularity

# Feature engineering configuration
feature_engineering:
  lookback: 3                # Lag feature steps
  roll_windows: [3, 6, 12]   # Rolling window sizes
  include_time_features: true  # Whether to include time features

# Prediction configuration
prediction:
  horizons_hourly: [1, 6, 12, 24]   # Hourly prediction time horizons
  horizons_daily: [1, 2, 3, 7]      # Daily prediction time horizons

# Classification task configuration
classification:
  base_target: "CO(GT)"      # Base target for classification task
  horizons: [1, 6, 12, 24]   # Prediction time points

# Training configuration
training:
  seed: 42                   # Random seed
  device: "mps"              # Computing device (cpu/cuda/mps)
  save_model: true           # Whether to save models
```

---

## Model Description

### Regression Models

Using **MultiOutputRegressor** to simultaneously predict concentrations of multiple pollutants:

1. **Linear Regression**: Baseline model
2. **Random Forest Regressor**: Ensemble learning
   - Hyperparameter tuning: n_estimators, max_depth, min_samples_split
3. **MLP Regressor**: Neural network (Multi-layer Perceptron)
   - Hyperparameter tuning: hidden_layer_sizes, max_iter, learning_rate_init

### Classification Models

Predicting future pollution levels (low/mid/high):

1. **Logistic Regression**
   - Hyperparameter: C (regularization strength)
2. **Random Forest Classifier**
3. **MLP Classifier**

Each model is trained for different prediction time points (t+1, t+6, t+12, t+24).

---

## Feature Engineering

### Lag Features
- Use pollutant concentrations from past 1-3 time steps as features

### Rolling Statistics Features
- Sliding window statistics: mean, standard deviation, maximum, minimum
- Window sizes: 3, 6, 12 (hours/days)

### Time Features
- Extract time information: hour, weekday, month, etc.
- Optional cyclical encoding (sine/cosine transformation)

### Target Variables
- **Regression Task**: Current time step pollutant concentrations (multi-output)
- **Classification Task**: Future t+h time step pollution levels (low/mid/high)

---

## Data Splitting

### Regression Task
- Training Set: 2004 data
- Validation Set: Last 10% of training set
- Test Set: 2005 data

### Classification Task
- Training Set: First 90% of 2004 data
- Validation Set: Last 10% of 2004 data
- Test Set: 2005 data (not used for training)

---

## Evaluation Metrics

### Regression Task
- **MAE** (Mean Absolute Error): Average absolute error
- **RMSE** (Root Mean Squared Error): Root mean squared error
- **R²** (R-squared): Coefficient of determination

### Classification Task
- **Accuracy**: Classification accuracy
- **Precision**: Precision (weighted average)
- **Recall**: Recall (weighted average)
- **F1-Score**: F1 score (weighted average)

### Also includes:
- Confusion Matrix
- Predicted vs. actual time series comparison plots

---

## Output Results

### Regression Task Output

For each time granularity (hourly/daily):
- `best_models/`: Best model files (.joblib format)
- `plots/`: Evaluation plots for each pollutant and model
  - Scatter plots (predicted vs. actual)
  - Time series comparison plots
  - Residual histograms
  - Residual time series plots
- `regression_val_metrics.csv`: Validation set metrics
- `test_metrics.csv`: Test set metrics

### Classification Task Output

For each time point (t+1, t+6, t+12, t+24):
- Best model files (.joblib format)
- `plots/`: Evaluation plots
  - Confusion matrices
  - Predicted vs. actual time series comparison plots
- `metrics_t+{horizon}.csv`: Detailed metrics
- `classification_val_summary.csv`: Validation set summary
- `classification_test_summary.csv`: Test set summary

---

## Visualization Features

1. **Pollutant Time Series**: Display trends of various pollutants over time
2. **Correlation Heatmap**: Correlation analysis between pollutants
3. **Meteorological-Pollutant Relationships**: Scatter plots showing relationships between temperature, humidity, absolute humidity and pollutants
4. **Time Pattern Analysis**: Periodic patterns for hour, week, and month
5. **Anomaly Detection**: Identify and visualize anomalous data points
6. **Model Performance Comparison**: Performance comparison of different models at different time points


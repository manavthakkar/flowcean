# ML Pipeline

This README describes the complete workflow of the `ml_pipeline`, from raw ROS2 bag processing to model training, and evaluation.

The pipeline consists of:

1. Dataset processing
2. Dataset building
3. Model training
4. Evaluation
5. Cleaning utilities
6. Configuration management

# Dataset Processing

## `process_bags.py`

### Purpose

Processes raw ROS2 bag files and converts them into structured tabular datasets (`.parquet` and `.csv`).

### What it does

* Reads bag paths from `config.yaml → rosbag.bags`
* Extracts configured ROS message topics
* Applies localization thresholds:

  * `position_threshold`
  * `heading_threshold`
* Creates labeled dataset (`is_delocalized`)
* Saves processed output to:

```
artifacts/datasets/processed/<bag_name>.parquet
artifacts/datasets/processed/<bag_name>.csv
```

### Run Command

```bash
python3 -m ml_pipeline.dataset.process_bags
```

---

## `build_experiment_dataset.py`

### Purpose

Builds the final **train** and **evaluation** datasets from processed bag files.

### What it does

* Loads processed `.parquet` files
* Concatenates them
* Sorts by time
* Creates:

```
artifacts/datasets/train.parquet
artifacts/datasets/eval.parquet
```

Paths are defined in:

```
config.yaml → rosbag.training_paths
config.yaml → rosbag.evaluation_paths
```

### Run Command

```bash
python3 -m ml_pipeline.dataset.build_experiment_dataset
```

# Model Training

All training scripts use:

* Optuna hyperparameter search
* Automatic evaluation
* Threshold sweep
* Plot generation
* PDF report generation (optional)
* Weights & Biases logging
* Experiment logging (CSV)

## `run.py`

### Purpose

Multi-algorithm training pipeline with Optuna.

### Supported Algorithms

* LightGBM
* Random Forest
* ExtraTrees
* CatBoost
* XGBoost

Optuna selects the best-performing algorithm automatically.

### What it does

1. Loads `train.parquet`
2. Applies feature toggles from `config.yaml`
3. Performs Optuna search
4. Trains best model on full dataset
5. Runs automatic evaluation on `eval.parquet`
6. Performs threshold sweep
7. Saves:

   * model.pkl
   * scaler.pkl
   * feature_columns.json
   * metadata.json
   * eval_results.parquet
8. Generates plots
9. Logs experiment to `experiment_log.csv`

### Run Command

```bash
python3 -m ml_pipeline.training.run
```

## `run_adaboost.py`

### Purpose

Optuna-based training specifically for **AdaBoost**.

### Key Features

* Uses `DecisionTree` as base estimator
* Handles class imbalance via:

  ```
  compute_sample_weight(class_weight="balanced")
  ```
* Automatic threshold sweep
* PDF report generation
* W&B logging

### Run Command

```bash
python3 -m ml_pipeline.training.run_adaboost
```

---

## `run_catboost.py`

### Purpose

Optuna-based training specifically for **CatBoost**.

### Key Features

* Native CatBoost hyperparameters
* Uses class weights `[w0, w1]`
* Automatic evaluation
* Threshold sweep
* Plot + report generation
* W&B logging

### Run Command

```bash
python3 -m ml_pipeline.training.run_catboost
```

# Model Evaluation

## `evaluate_model.py`

### Purpose

Evaluate a trained model on the evaluation dataset.

### What it does

* Loads selected model from `artifacts/models/`
* Loads `eval.parquet`
* Applies temporal features if required
* Applies scaling
* Predicts probabilities
* Applies decision threshold
* Optional temporal smoothing
* Prints:

  * Precision
  * Recall
  * F1
  * F0.5
  * Confusion matrix

### Run Commands

Interactive model selection:

```bash
python3 -m ml_pipeline.evaluation.evaluate_model
```

Specify model and threshold:

```bash
python3 -m ml_pipeline.evaluation.evaluate_model --model_dir <model_dir> --threshold 0.50
```

# Cleaning Utilities

## `clean_dataset.py`

### Purpose

Removes generated dataset files.

### Run Command

```bash
python3 -m ml_pipeline.clean_dataset
```

## `clean_models.py`

### Purpose

Removes trained model directories from `artifacts/models`.

### Run Command

```bash
python3 -m ml_pipeline.clean_models
```

# Configuration


## `config.yaml`

### Purpose

Central configuration file controlling the entire pipeline.

## Sections Explained

### `rosbag`

Defines:

* `training_paths`
* `evaluation_paths`
* `bags`
* `message_paths`

---

### `features`

Controls feature engineering:

```yaml
features:
  use_temporal: false
  use_scanmap: true
  use_particle: true
  use_amcl_pose: false
```

---

### `experiment`

Defines metadata:

* train_maps
* eval_maps
* odometry
* notes
* tags

Used for:

* W&B logging
* Experiment CSV log
* PDF report

---

### `localization`

Defines labeling thresholds:

```yaml
localization:
  position_threshold: 0.2
  heading_threshold: 0.2
```

These thresholds define when a sample is labeled as `is_delocalized = 1`.

---

### `optuna`

Defines training parameters:

```yaml
optuna:
  n_trials: 100
  model_name: "small_odometry_drift"
  early_stop_f1: 0.95
```

---

# Full Pipeline Execution Order

```bash
# 1. Process raw bags
python3 -m ml_pipeline.dataset.process_bags

# 2. Build train/eval datasets
python3 -m ml_pipeline.dataset.build_experiment_dataset

# 3. Train model
python3 -m ml_pipeline.training.run
# OR
python3 -m ml_pipeline.training.run_adaboost
python3 -m ml_pipeline.training.run_catboost

# 4. Evaluate model
python3 -m ml_pipeline.evaluation.evaluate_model
python3 -m ml_pipeline.evaluation.evaluate_model --model_dir <model_dir> --threshold 0.50

# 5. Clean if needed
python3 -m ml_pipeline.clean_dataset
python3 -m ml_pipeline.clean_models
```

---

# Artifacts Structure

```
artifacts/
│
├── datasets/
│   ├── processed/
│   ├── train.parquet
│   └── eval.parquet
│
├── models/
│   └── <model_name_timestamp>/
│       ├── model.pkl
│       ├── scaler.pkl
│       ├── feature_columns.json
│       ├── metadata.json
│       └── eval_results.parquet
│
└── experiment_log.csv
```

---

# Summary

This pipeline provides:

* Reproducible dataset generation
* Config-driven feature control
* Automated hyperparameter optimization
* Built-in class imbalance handling
* Threshold optimization
* Experiment tracking
* Automatic plotting
* PDF reporting
* W&B integration
* Clean experiment logging

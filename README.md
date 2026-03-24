# Multi-Label Email Classification (Chained Multi-Output Architecture)

The implementations in this repository primarily focus on the **Chained Multi-Output Architecture** for multi-label email classification. In this approach, the machine learning model predicts the dependent variables sequentially, where each prediction depends on the previous one:

1. **First:** Type 2
2. **Second:** Type 2 + Type 3
3. **Third:** Type 2 + Type 3 + Type 4

This design captures the dependencies between labels and ensures that the predictions follow a structured chained structure.

---

## Project Directory Structure

```text
├── main.py                          # Main entry point that coordinates the full workflow
├── Config.py                        # Configuration file
│
├── preprocessing/
│   ├── __init__.py                  # Preprocessing package entry
│   └── pipeline.py                  # Data loading, cleaning, and chained label construction
│
├── data/
│   ├── __init__.py                  # Data package entry
│   ├── dataset.py                   # Dataset preparation and train/test split logic
│   ├── AppGallery.csv               # Input dataset 1
│   └── Purchasing.csv               # Input dataset 2
│
├── modelling/
│   ├── __init__.py                  # Modelling package entry
│   ├── runner.py                    # Model runner and evaluation controller
│   └── results.py                   # Result aggregation and CSV export
│
├── models/                          # Model implementation directory
│   ├── base.py                      # Shared base model interface
│   ├── random_forest_model.py       # Random Forest classifier
│   ├── hist_gb_model.py             # Histogram Gradient Boosting classifier
│   ├── sgd_model.py                 # SGD classifier
│   ├── adaboost_model.py            # AdaBoost classifier
│   ├── voting_model.py              # Voting ensemble classifier
│   └── extra_trees_model.py         # Extra Trees classifier
│
├── cleaned_tickets.csv              # Cleaned intermediate dataset
└── results_summary.csv              # Final result output



## Execution Flow

```text
main.py (entry point)
  ↓
prepare_data() [preprocessing/pipeline.py]
  → Read data/AppGallery.csv + data/Purchasing.csv
  → Apply column mapping and text cleaning
  → Build chained labels
  → Save cleaned_tickets.csv
  ↓
build_dataset_bundle() [data/dataset.py]
  → Vectorize text with TF-IDF
  → Split data into training and testing sets (80/20)
  → Build aligned chained datasets
  ↓
ModelRunner.run() [modelling/runner.py]
  → Train 6 models in sequence:
    - RandomForest
    - HistGradientBoosting
    - SGD
    - AdaBoost
    - Voting
    - ExtraTrees
  → Evaluate each model across 3 chained levels
  → Save results_summary.csv

## Design Patterns 

Facade/Controller (main.py)
main.py orchestrates preprocessing → dataset building → modelling → results export.
Keeps business logic in modules, main is a single coordinator.

Strategy (models)
RandomForest, HistGradientBoosting, SGD, AdaBoost, Voting, ExtraTrees implement a common model interface.
Runner can swap algorithms without pipeline changes.

Factory/Registry (model registration)
Central model registry maps names to classes; runner instantiates models dynamically.
Add new model with minimal pipeline edits.

Template Method (BaseModel)
Base class defines training/predict/report workflow; concrete models implement estimator-specific details.
Reduces duplicate code and standardizes behavior.

Data Transfer Objects (data containers)
Structured objects for prepared data, dataset bundle, level results, model results.
Ensures clear, consistent multi-level chain label flow.

## Environment Setup

Follow the steps below to set up the environment using Conda:

```bash
conda create -n engineering python=3.10 -y
conda activate engineering
```

## Install Dependencies

Install the required Python libraries:

```bash
conda install pandas
conda install scikit-learn
```

## Navigate to Project Directory

Install the required Python libraries:

```bash
cd /path/to/your/project
```

> Please replace /path/to/your/project with the path to your project on your computer, for example:
> 
> - **macOS:** /Users/yourname/Desktop/your_project
> - **Windows:** C:\Users\yourname\Desktop\your_project

## Run the Project

Execute the main script:

```bash
python main.py
```

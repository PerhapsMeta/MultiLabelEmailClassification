# Multi-Label Email Classification (Chained Multi-Output Architecture)

The implementations in this repository primarily focus on the **Chained Multi-Output Architecture** for multi-label email classification. In this approach, the machine learning model predicts the dependent variables sequentially, where each prediction depends on the previous one:

1. **First:** Type 2
2. **Second:** Type 2 + Type 3
3. **Third:** Type 2 + Type 3 + Type 4

This design captures the dependencies between labels and ensures that the predictions follow a structured chained structure.

---

## Result Sample

| model_name   | level_name                              | level_accuracy | model_accuracy |
|--------------|------------------------------------------|--------------|---------------------|
| RandomForest | Level 1 (Type 2)                         | 0.775        | 0.7083333333333334  |
| RandomForest | Level 2 (Type 2 + Type 3)                | 0.7          | 0.7083333333333334  |
| RandomForest | Level 3 (Type 2 + Type 3 + Type 4)       | 0.65         | 0.7083333333333334  |

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
```


## Execution Flow

```text
+----------------------+
|   main.py            |
|   (Entry Point)      |
+----------+-----------+
           |
           v
+----------------------+
| prepare_data()       |
| preprocessing/       |
| pipeline.py          |
+----------+-----------+
           |
           +--> Read AppGallery.csv
           |
           +--> Read Purchasing.csv
           |
           +--> Column mapping
           |
           +--> Text cleaning
           |
           +--> Build chained labels
           |
           +--> Save cleaned_tickets.csv
           |
           v
+----------------------+
| build_dataset_bundle() |
| data/dataset.py        |
+----------+-------------+
           |
           +--> TF-IDF vectorization
           |
           +--> Train/Test split
           |    (80/20)
           |
           +--> Build aligned
           |    chained datasets
           |
           v
+----------------------+
| ModelRunner.run()    |
| modelling/runner.py  |
+----------+-----------+
           |
           +--> Train models
           |    - RandomForest
           |    - HistGradientBoosting
           |    - SGD
           |    - AdaBoost
           |    - Voting
           |    - ExtraTrees
           |
           +--> Evaluate 3 levels
           |    - Level 1 (Type 2)
           |    - Level 2 (Type 2 + Type 3)
           |    - Level 3 (Type 2 + Type 3 + Type 4)
           |
           +--> Save results_summary.csv
```

## Design Patterns 


- **Facade / Controller (main.py)**
  - `main.py` orchestrates preprocessing → dataset building → modelling → results export  
  - Keeps business logic in modules; main acts as a single coordinator

- **Strategy Pattern (models)**
  - RandomForest, HistGradientBoosting, SGD, AdaBoost, Voting, ExtraTrees implement a common model interface  
  - Runner can swap algorithms without changing the pipeline

- **Factory / Registry (model registration)**
  - Central model registry maps model names to classes  
  - Runner dynamically instantiates models  
  - New models can be added with minimal pipeline modification

- **Template Method (BaseModel)**
  - Base class defines the workflow: train → predict → report  
  - Concrete models implement estimator-specific details  
  - Reduces code duplication and enforces standard behavior

- **Data Transfer Objects (DTOs / data containers)**
  - Structured objects for:
    - prepared data  
    - dataset bundle  
    - level results  
    - model results  
  - Ensures consistent data flow across multi-level chained labels

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

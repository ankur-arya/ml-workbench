# Machine Learning Toolkit

This repository contains two interactive web applications built with Streamlit for machine learning experimentation:

1. Dataset Generator (`app.py`)
2. Model Trainer (`model_trainer.py`)

## Dataset Generator Features

- Generate datasets for different ML tasks:
  - Classification (make_classification, make_moons, make_circles)
  - Regression
  - Clustering
  - Dimensionality Reduction (Swiss Roll)
- Interactive parameter selection
- 2D visualization of generated datasets
- PCA visualization for high-dimensional data
- Feature selection for visualization

## Model Trainer Features

- Compatible with all dataset types from Dataset Generator
- Supports multiple machine learning tasks:
  - Classification (Logistic Regression, Random Forest, SVM)
  - Regression (Linear Regression, Random Forest, SVR, Ridge, Lasso)
  - Clustering (K-Means, DBSCAN, Agglomerative)
  - Dimensionality Reduction (PCA, T-SNE, TruncatedSVD)
- Automatic validation of dataset-task-algorithm compatibility
- Multiple evaluation metrics for each task type
- Training progress visualization with incremental data sizes
- Train/Test split evaluation (80/20)
- Interactive model parameter selection
- Real-time training progress tracking
- Comprehensive performance metrics visualization

## Setup

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Applications

To run the Dataset Generator:
```bash
streamlit run app.py
```

To run the Model Trainer:
```bash
streamlit run model_trainer.py
```

## Usage

### Dataset Generator
1. Select the type of ML task
2. Adjust the parameters according to your needs
3. Generate and visualize the dataset
4. For high-dimensional data:
   - Select specific features to plot
   - Optionally use PCA for visualization

### Model Trainer
1. Generate a dataset using the available options
2. Select a compatible machine learning task
3. Choose a model from the available options
4. Select evaluation metrics
5. Train the model and monitor progress
6. View final performance metrics on train and test sets

## Requirements

- Python 3.7+
- streamlit
- scikit-learn
- numpy
- pandas
- matplotlib # ml-workbench

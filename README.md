# LendingClub Loan Default Prediction

## Project Overview
This project explores machine learning techniques for predicting loan repayment outcomes using historical LendingClub loan data.

The project is based on an existing research paper and its accompanying source code. Since the original dataset was not available in the same form, a substitute LendingClub dataset was used. Therefore, this is a reproduction attempt rather than an exact replication.

## Authors
- Amulya P
- Zayaan Ahmed

## Machine Learning Models

### Classification
- Logistic Regression
- Random Forest Classifier
- Neural Network (MLP Classifier)

The classification task distinguishes fully paid loans from charged-off or defaulted loans, according to the available dataset labels.

### Regression
The project also explores regression approaches for predicting loan returns:
- Linear Regression
- Ridge Regression
- Neural Network Regressor
- Random Forest Regressor

## Dataset
The project uses the LendingClub `accepted_2007_to_2018Q4.csv.gz` dataset.

The raw dataset is not included in this repository. It must be obtained separately, and the dataset path must be configured before running the notebook.

## Repository Structure

```text
LendingClub-Loan-Default-Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── LendingClub_Phase1.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── classification.py
│   ├── regression.py
│   └── evaluation.py
├── results/
│   └── figures/
├── reports/
└── original_code/
    └── README.md
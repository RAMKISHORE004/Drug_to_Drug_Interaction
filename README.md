# Drug_to_Drug_Interaction

## Overview:

This repository contains the implementation of GNN-DDI, a deep learning-based model for predicting drug-drug interaction (DDI) events using graph neural networks (GNNs). In this project we implement Drug-Drug Interaction (DDI) Prediction using various machine learning and deep learning models. It applies data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation to classify drug interactions. The model integrates drug features from various sources into an attributed heterogeneous network and applies deep learning techniques for event classification.

## Dataset:

The dataset DDICorpus2013 is sourced from DrugBank.

![image](https://github.com/user-attachments/assets/8003d8c7-9d58-46ff-babe-7838b5583cbf)


## Features:

Data Preprocessing: Cleaning, normalizing, and feature engineering on drug interaction data.

Baseline Machine Learning Models: Logistic Regression, Naïve Bayes, Random Forest.

Deep Learning Model: BioBERT, a transformer-based model for DDI classification.

Graph Neural Networks (GNNs): for feature extraction from an attributed heterogeneous network.

Class Imbalance Handling: Generating synthetic negative samples.

Hyperparameter Tuning: Cross-validation and model fine-tuning.

Evaluation Metrics: Accuracy, Precision, Recall, F1-score.

## Installation & Setup
### Clone the Repository

git clone https://github.com/RAMKISHORE004/Drug_to_Drug_Interaction.git

cd Drug_to_Drug_Interaction

### Install Dependencies

Create a virtual environment (optional but recommended):

python -m venv venv

source venv/bin/activate   # For macOS/Linux

venv\Scripts\activate      # For Windows

Then install dependencies:

pip install -r requirements.txt

# Model Selection & Implementation

The project implements multiple models for predicting drug-drug interactions (DDIs).

## Machine Learning Models

Logistic Regression:	Baseline model with L2 regularization

Naïve Bayes:	Probability-based model (MultinomialNB)

Random Forest:	Ensemble learning with decision trees

### Synthetic Data Generation

Randomly generate non-interacting drug pairs to balance classes.

Helps reduce model bias towards positive interactions.

### Hyperparameter Tuning & Cross-Validation

Stratified K-Fold Cross-Validation (K=5)

### Fine-tuned parameters:

Logistic Regression (C=0.1, max_iter=300)

Naïve Bayes (alpha=0.5)

Random Forest (n_estimators=100, max_depth=10)

Deep Learning Model: BioBERT

Model: dmis-lab/biobert-base-cased-v1.1

Tokenizer: BERT Tokenizer

### Training Method:

Dataset tokenized with BERT Tokenizer

Fine-tuned on labeled drug interaction sentences

Optimized using Adam optimizer (learning_rate=2e-5)

## Model Evaluation

Metrics used are 
Accuracy,
Precision,
Recall,
F1-score,
AUC-ROC Curve


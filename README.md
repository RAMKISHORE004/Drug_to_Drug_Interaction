# Drug_to_Drug_Interaction

## Overview:

This repository contains the implementation of GNN-DDI, a deep learning-based model for predicting drug-drug interaction (DDI) events using graph neural networks (GNNs). In this project we implement Drug-Drug Interaction (DDI) Prediction using various machine learning and deep learning models. It applies data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation to classify drug interactions. The model integrates drug features from various sources into an attributed heterogeneous network and applies deep learning techniques for event classification.
## Repository Structure
├── train_model.py          # Training script for GAT + edge‑classifier
├── app.py                  # Flask app for serving predictions
├── index.html              # Frontend interface
├── DDICorpus2013.csv       # Dataset of drug pairs and sentences
├── gat_best.pth            # Saved GAT model weights
├── edge_best.pth           # Saved edge‑classifier weights
├── requirements.txt        # Python dependencies
└── README.md               # This file

## Dataset:

The dataset DDICorpus2013 is sourced from DrugBank.

![image](https://github.com/user-attachments/assets/8003d8c7-9d58-46ff-babe-7838b5583cbf)


## Features:

Data Preprocessing: Cleaning, normalizing, and feature engineering on drug interaction data.

Baseline Machine Learning Models: Logistic Regression, Naïve Bayes, Random Forest.

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


### Fine-tuned parameters:

Logistic Regression (C=0.1, max_iter=300)

Naïve Bayes (alpha=0.5)

Random Forest (n_estimators=100, max_depth=10)


### Training the model:

py train_model.py

The script will:

Load and preprocess the data

Build negative samples to balance the dataset

Construct the graph and TF–IDF features

Train a two‐layer GAT and an edge MLP classifier for 20 epochs

Save gat_best.pth and edge_best.pth when validation accuracy improves

Plot and display loss & accuracy curves

## Serving Predictions

Make sure gat_best.pth, edge_best.pth, index.html, and DDICorpus2013.csv are present.

Start the Flask server:
py app.py

## Usage
Frontend: Enter 2–4 comma‐separated drug names and click "Check Interactions". The app will display risk‐level summaries.

API: Send a POST request to /predict with JSON payload

## Code Highlights

Graph Construction: Drugs are nodes, TF–IDF sentences are edge features, and negative sampling balances positives and negatives.

GAT: Two GATConv layers learn 128‐dim embeddings with multi‐head attention.

Edge Classifier: A two‐layer MLP combines the embeddings and TF–IDF vector to predict interaction probability.

Templates: Probability buckets (high/medium/low) map to human‐readable sentence templates.

## Future Work

Integrate BioBERT features instead of TF–IDF for stronger semantic signals.

Add user feedback capture in the frontend for iterative retraining.

Deploy to production (Docker, Kubernetes) for scalable inference.

## Workflow 
```mermaid
flowchart TB
  subgraph Data_Preparation
    A1[Read DDICorpus2013.csv]
    A2[Deduplicate & Clean Text]
    A3[Normalize Drug Names]
    A4[Fit TF–IDF on Sentences]
  end

  subgraph Negative_Sampling
    B1[List All Drug Pairs]
    B2[Sample Equal Negatives]
    B3[Zero‑Pad TF–IDF for Negatives]
  end

  subgraph Feature_Construction
    C1[Build One‑Hot Node Matrix]
    C2[Extract TF–IDF Edge Features]
  end

  subgraph Graph_Assembly
    D1[Make Bidirectional Edge List]
    D2[Convert to 2×E edge_index]
    D3[Wrap into PyG Data Object]
  end

  subgraph Model_Training
    E1[Run GATConv Layer1 4 heads + ELU]
    E2[Run GATConv Layer2 1 head]
    E3[Train Edge MLP 356→64→1 + Sigmoid]
    E4[Optimize with Adam & BCELoss]
    E5[Save gat_best.pth & edge_best.pth]
  end

  subgraph Inference_Deployment
    F1[Load Models & TF–IDF Vectorizer]
    F2[Precompute Node Embeddings]
    F3[Concatenate Embeddings + TF–IDF]
    F4[Predict Interaction Probability]
    F5[Bucket into High/Medium/Low]
    F6[Fill Sentence Templates]
    F7[Serve via Flask & HTML/JS Frontend]
  end

  Data_Preparation --> Negative_Sampling
  Negative_Sampling --> Feature_Construction
  Feature_Construction --> Graph_Assembly
  Graph_Assembly --> Model_Training
  Model_Training --> Inference_Deployment










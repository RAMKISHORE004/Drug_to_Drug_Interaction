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
flowchart LR
  subgraph Data_Prep["Data Preparation"]
    A1[Load DDICorpus2013.csv]
    A2[Drop duplicates & clean text]
    A3[Normalize drug names]
    A4[TF–IDF fit on sentences]
  end

  subgraph Sampling["Negative Sampling"]
    B1[Enumerate all drug pairs]
    B2[Select equal number of non‑interactions]
    B3[Assign zero TF–IDF vectors]
  end

  subgraph Features["Feature Construction"]
    C1[Create one‑hot node matrix]
    C2[Use TF–IDF for edge features]
  end

  subgraph Graph["Graph Assembly"]
    D1[Build bidirectional edge_list]
    D2[Convert to edge_index tensor]
    D3[Package into PyG Data]
  end

  subgraph Model["Model Training"]
    E1[GAT Layer 1: 4 heads → 512 dim + ELU]
    E2[GAT Layer 2: 1 head → 128 dim]
    E3[Edge MLP: 256+100→64→1 + Sigmoid]
    E4[Train with Adam & BCELoss]
    E5[Save best weights]
  end

  subgraph Inference["Inference & Templating"]
    F1[Load saved weights & TF–IDF]
    F2[Precompute node embeddings]
    F3[For each drug pair: concat embeddings + TF–IDF]
    F4[Predict probability p]
    F5[Bucket p → High/Med/Low]
    F6[Select and fill template]
  end

  subgraph Deploy["Deployment"]
    G1[Flask: serve index.html]
    G2[Flask API: POST /predict]
    G3[HTML/JS frontend]
  end

  Data_Prep --> Sampling
  Sampling --> Features
  Features --> Graph
  Graph --> Model
  Model --> Inference
  Inference --> Deploy











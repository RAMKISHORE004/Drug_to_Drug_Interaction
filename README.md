
# Drug-Drug Interaction Prediction

This repository contains a Graph Attention Network (GAT)-based model for predicting interactions between drugs. The goal is to identify potential adverse interactions between drug pairs using graph-based machine learning techniques.

## 🚀 Features
- **Graph Neural Network**: Uses a Graph Attention Network to model drug relationships.
- **High Accuracy**: Trained on real-world drug interaction datasets.
- **Web Interface**: User-friendly web app to interact with the model and make predictions.

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/drug-drug-interaction.git
   cd drug-drug-interaction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:8000`.

## 🧠 Model Training

To train the model from scratch:
```bash
python train.py --dataset <path_to_dataset>
```

## 🔍 Predicting Interactions

To predict interaction between two drugs:
```bash
python predict.py --input <drug1> <drug2>
```

## 🌐 Web Interface

You can also use the built-in web interface to:
- Input any two drug names.
- Get real-time predictions.
- Visualize attention weights (optional feature).

## 📊 Dataset

The model is trained on a benchmark drug-drug interaction dataset. Preprocessing scripts are included to:
- Clean and structure raw data.
- Convert drug interactions into graph format.
- Split the data into training/validation/test sets.

## 🤝 Contributing

Contributions are welcome! If you have suggestions, feel free to:
- Fork the repo
- Create a feature branch
- Open a Pull Request

Please make sure your code adheres to the project's style and passes tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- GAT architecture inspired by [Velickovic et al., 2018](https://arxiv.org/abs/1710.10903)
- Dataset from DrugBank / TWOSIDES / other sources
- Special thanks to contributors and reviewers!

---

🧪 *Note: This model is for research purposes only and should not be used in clinical settings without professional validation.*
```









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













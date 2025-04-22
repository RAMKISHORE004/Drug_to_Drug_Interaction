
# Drug-Drug Interaction Prediction

This repository contains a Graph Attention Network (GAT)-based model for predicting interactions between drugs. The goal is to identify potential adverse interactions between drug pairs using graph-based machine learning techniques.

## 🚀 Features
- **Graph Neural Network**: Uses a Graph Attention Network to model drug relationships.
- **High Accuracy**: Trained on real-world drug interaction datasets.
- **Web Interface**: User-friendly web app to interact with the model and make predictions.

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/RAMKISHORE004/Drug_to_Drug_Interaction.git
   cd Drug_to_Drug_Interaction
   ```
2. **Create a Virtual Environment**
   ```bash
   #Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate   # For macOS/Linux
   venv\Scripts\activate      # For Windows
    ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the web application**
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:5000`.

## 🧠 Model Training

To train the model from scratch:
```bash
python train.py --dataset DDICorpus2013.xlsx
```

## 🧠 Model Selection & Implementation

This project implements multiple models for predicting drug-drug interactions (DDIs):

### Machine Learning Models
- **Logistic Regression**: Baseline model with L2 regularization  
- **Naïve Bayes**: Probability-based model (`MultinomialNB`)  
- **Random Forest**: Ensemble learning with decision trees  

### Synthetic Data Generation

Randomly generate non-interacting drug pairs to balance classes.  
Helps reduce model bias towards positive interactions.

### Fine-tuned Parameters:
- Logistic Regression: `C=0.1`, `max_iter=300`
- Naïve Bayes: `alpha=0.5`
- Random Forest: `n_estimators=100`, `max_depth=10`

## 🌐 Web Interface

You can also use the built-in web interface to:
- Input any 2 to 4 drug names.
- Get real-time predictions.
- Visualize attention weights (optional feature).

## 📊 Dataset

The model is trained on a benchmark drug-drug interaction dataset. Preprocessing scripts are included to:
- Clean and structure raw data.
- Convert drug interactions into graph format.
- Split the data into training/validation/test sets.
- ![image](https://github.com/user-attachments/assets/8003d8c7-9d58-46ff-babe-7838b5583cbf)

## 🤝 Contributing

Contributions are welcome! If you have suggestions, feel free to:
- Fork the repo
- Create a feature branch
- Open a Pull Request

Please make sure your code adheres to the project's style and passes tests.

## 🧭 Usage

**Frontend**: Enter 2–4 comma‐separated drug names and click "Check Interactions". 

The app will display risk‐level summaries.

## 🧩 Code Highlights

- **Graph Construction**: Drugs are nodes, TF–IDF sentences are edge features, and negative sampling balances positives and negatives.
- **GAT**: Two `GATConv` layers learn 128‐dim embeddings with multi‐head attention.
- **Edge Classifier**: A two‐layer MLP combines the embeddings and TF–IDF vector to predict interaction probability.
- **Templates**: Probability buckets (high/medium/low) map to human‐readable sentence templates.

## 🛠️ Future Work

- Integrate BioBERT features instead of TF–IDF for stronger semantic signals.
- Add user feedback capture in the frontend for iterative retraining.
- Deploy to production (Docker, Kubernetes) for scalable inference.



## 🙏 Acknowledgments

- GAT architecture inspired by [Velickovic et al., 2018](https://arxiv.org/abs/1710.10903)
- Dataset from DrugBank / TWOSIDES / other sources
- Special thanks to contributors and reviewers!

---

🧪 *Note: This model is for research purposes only and should not be used in clinical settings without professional validation.*

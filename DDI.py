#Drug to Drug Interaction code
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
# from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
# Load the dataset
file_path = "C:/Users/ahana/OneDrive/Desktop/capstone/Drug_to_Drug_Interaction/DDICorpus2013.xlsx"
df = pd.read_excel(file_path)
print(df.head(10))
print(df.shape)
 #Check for duplicate rows
print(f"Number of duplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(df.shape)



# Step 1: Handle Missing Values (Remove rows with missing drug names or interaction sentences)
df_preprocessed = df.dropna(subset=["Drug_1_Name", "Drug_2_Name", "Sentence_Text"])

# Step 2: Normalize Drug Names (Lowercase, strip spaces)
df_preprocessed["Drug_1_Name"] = df_preprocessed["Drug_1_Name"].str.lower().str.strip()
df_preprocessed["Drug_2_Name"] = df_preprocessed["Drug_2_Name"].str.lower().str.strip()

# Step 3: Expand Drug Synonyms (Mapping common synonyms)
synonym_map = {
    "aspirin": "acetylsalicylic acid",
    "paracetamol": "acetaminophen",
    "tylenol": "acetaminophen",
    "ibuprofen": "advil",
    "omeprazole": "prilosec",
}




# Get unique drug names from both columns
unique_drugs = list(set(df_preprocessed["Drug_1_Name"]).union(set(df_preprocessed["Drug_2_Name"])))

# Generate synthetic non-interacting pairs
num_neg_samples = len(df_preprocessed)  # Generate as many negatives as positives
negative_samples = []

while len(negative_samples) < num_neg_samples:
    drug1, drug2 = random.sample(unique_drugs, 2)  # Randomly pick two drugs
    # Ensure the pair does not exist in the original dataset
    if not (
        ((df_preprocessed["Drug_1_Name"] == drug1) & (df_preprocessed["Drug_2_Name"] == drug2)).any() or
        ((df_preprocessed["Drug_1_Name"] == drug2) & (df_preprocessed["Drug_2_Name"] == drug1)).any()
    ):
        negative_samples.append([drug1, drug2, "No known interaction.", 0])

# Convert to DataFrame and append to existing data
df_negative = pd.DataFrame(negative_samples, columns=["Drug_1_Name", "Drug_2_Name", "Sentence_Text", "Is_DDI"])
df_balanced = pd.concat([df_preprocessed, df_negative], ignore_index=True)

# Re-run TF-IDF vectorization
X_tfidf_balanced = vectorizer.fit_transform(df_balanced["Sentence_Text"])

# Display balanced dataset
df_balanced.tail(10)

negative_samples_improved = 1


# Step 6: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_tfidf_balanced = vectorizer.fit_transform(df_balanced["Sentence_Text"])


df_preprocessed["Is_DDI"] = df_preprocessed["Is_DDI"].astype(int)

if df_preprocessed["Is_DDI"].nunique() < 2:  # to check if there are 2 classes present for logistic regression
    print("Warning: Target variable has only one class. Logistic Regression might fail.")

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf_balanced, df_preprocessed["Is_DDI"], test_size=0.2, random_state=42, stratify=df_preprocessed["Is_DDI"]
)


baseline_models = {   # model initialization
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}


model_results = {}
for name, model in baseline_models.items():  # training and evaluating models
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True),
    }



#Number of classes present in Is_DDI
class_distribution = df_preprocessed["Is_DDI"].value_counts(normalize=True)
class_distribution

import random


unique_drugs = list(set(df_preprocessed["Drug_1_Name"]).union(set(df_preprocessed["Drug_2_Name"])))  # Getting unique drug names from both columns


num_neg_samples = len(df_preprocessed)  # Generate synthetic non-interacting pairs in equal quantity as positive samples
negative_samples = []

while len(negative_samples) < num_neg_samples:
    drug1, drug2 = random.sample(unique_drugs, 2)  # Randomly picking two drugs

    if not (
        ((df_preprocessed["Drug_1_Name"] == drug1) & (df_preprocessed["Drug_2_Name"] == drug2)).any() or    # Ensuring the pair does not exist in the original dataset
        ((df_preprocessed["Drug_1_Name"] == drug2) & (df_preprocessed["Drug_2_Name"] == drug1)).any()
    ):
        negative_samples.append([drug1, drug2, "No known interaction.", 0])


df_negative = pd.DataFrame(negative_samples, columns=["Drug_1_Name", "Drug_2_Name", "Sentence_Text", "Is_DDI"])
df_balanced = pd.concat([df_preprocessed, df_negative], ignore_index=True)


X_tfidf_balanced = vectorizer.fit_transform(df_balanced["Sentence_Text"])  # TF-IDF vectorization


df_balanced.tail(10) # balanced dataset


X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    X_tfidf_balanced, df_balanced["Is_DDI"], test_size=0.2, random_state=42
)

baseline_models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

balanced_model_results = {}
for name, model in baseline_models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred_bal = model.predict(X_test_bal)
    balanced_model_results[name] = {
        "Accuracy": accuracy_score(y_test_bal, y_pred_bal),
        "Classification Report": classification_report(y_test_bal, y_pred_bal, output_dict=True),
    }

# Display model comparison results
result_df_balanced = pd.DataFrame({model: balanced_model_results[model]["Accuracy"] for model in balanced_model_results}, index=["Accuracy"])

{result_df_balanced.to_string(index=True)}





cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # cross-validation strategy


cv_results = {}

for name, model in baseline_models.items():  # performing K-Fold Cross-Validation and applying hyperparameter tuning
    scores = cross_val_score(model, X_tfidf_balanced, df_balanced["Is_DDI"], cv=cv, scoring='f1')
    cv_results[name] = {
        "Mean F1-Score": scores.mean(),
        "Standard Deviation": scores.std()
    }


cv_results_df = pd.DataFrame(cv_results).T

tuned_models = {
    "Logistic Regression": LogisticRegression(C=0.1, max_iter=300),  # Added regularization (C=0.1)
    "Naive Bayes": MultinomialNB(alpha=0.5),  # Laplace smoothing
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
}

tuned_model_results = {}
for name, model in tuned_models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred_tuned = model.predict(X_test_bal)
    tuned_model_results[name] = {
        "Accuracy": accuracy_score(y_test_bal, y_pred_tuned),
        "Classification Report": classification_report(y_test_bal, y_pred_tuned, output_dict=True),
    }
tuned_model_metrics = {}
for model_name, results in tuned_model_results.items():
    report = results["Classification Report"]
    tuned_model_metrics[model_name] = {
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-Score": report["1"]["f1-score"],
    }
tuned_metrics_df = pd.DataFrame(tuned_model_metrics).T


print("\n=== Cross-Validation Results (F1-Score) ===")
print(cv_results_df.to_string())
print("\n=== Tuned Model Performance Metrics ===")
print(tuned_metrics_df.to_string())


df_negative_improved = pd.DataFrame(negative_samples_improved, columns=["Drug_1_Name", "Drug_2_Name", "Sentence_Text", "Is_DDI"])
df_balanced_improved = pd.concat([df, df_negative_improved], ignore_index=True)

X_tfidf_balanced_improved = vectorizer.fit_transform(df_balanced_improved["Sentence_Text"])

X_train_bal_imp, X_test_bal_imp, y_train_bal_imp, y_test_bal_imp = train_test_split(
    X_tfidf_balanced_improved, df_balanced_improved["Is_DDI"], test_size=0.2, random_state=42
)

tuned_model_results_improved = {}
for name, model in tuned_models.items():
    model.fit(X_train_bal_imp, y_train_bal_imp)
    y_pred_tuned_imp = model.predict(X_test_bal_imp)
    tuned_model_results_improved[name] = {
        "Accuracy": accuracy_score(y_test_bal_imp, y_pred_tuned_imp),
        "Classification Report": classification_report(y_test_bal_imp, y_pred_tuned_imp, output_dict=True),
    }

tuned_model_metrics_improved = {}
for model_name, results in tuned_model_results_improved.items():
    report = results["Classification Report"]
    tuned_model_metrics_improved[model_name] = {
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-Score": report["1"]["f1-score"],
    }

tuned_metrics_df_improved = pd.DataFrame(tuned_model_metrics_improved).T

print("\n=== Improved Dataset ===")
print(df_balanced_improved.head())

print("\n=== Improved Model Performance Metrics ===")
print(tuned_metrics_df_improved.to_string())


print("\n=== Improved Balanced Dataset Sample (First 10 Rows) ===")
print(df_balanced_improved.head(10).to_string(index=False))

print("\n=== Improved Model Performance Metrics ===")
print(tuned_metrics_df_improved.to_string(index=True))





model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


class DrugInteractionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
X_train_texts, X_test_texts, y_train_labels, y_test_labels = train_test_split(
    df_balanced_improved["Sentence_Text"], df_balanced_improved["Is_DDI"], test_size=0.2, random_state=42
)

train_dataset = DrugInteractionDataset(X_train_texts.tolist(), y_train_labels.tolist(), tokenizer)
test_dataset = DrugInteractionDataset(X_test_texts.tolist(), y_test_labels.tolist(), tokenizer)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

eval_results = trainer.evaluate()

eval_results

test_loader = DataLoader(test_dataset, batch_size=8)
y_true = []
y_pred = []


model.eval()

for batch in test_loader:
    inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    y_true.extend(batch["labels"].numpy())
    y_pred.extend(predictions.cpu().numpy())


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

metrics_dict = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}

metrics_dict

def predict_with_biobert(trainer, dataset):
    predictions = trainer.predict(dataset)
    preds = predictions.predictions.argmax(axis=1)
    return preds
y_pred_biobert = predict_with_biobert(trainer, test_dataset)

bio_bert_report = classification_report(y_test_labels, y_pred_biobert, target_names=["Non-Interaction", "Interaction"], output_dict=True)

bio_bert_metrics = {
    "Precision": bio_bert_report["Interaction"]["precision"],
    "Recall": bio_bert_report["Interaction"]["recall"],
    "F1-Score": bio_bert_report["Interaction"]["f1-score"],
    "Accuracy": bio_bert_report["accuracy"],
}

bio_bert_metrics_df = pd.DataFrame([bio_bert_metrics], index=["BioBERT"])
bio_bert_metrics_df


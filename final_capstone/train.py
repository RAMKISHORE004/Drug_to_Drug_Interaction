# ----------------------------
# 1. Imports & Hyperparameters
# ----------------------------
import re
import random
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# File paths and model/training settings
CSV_PATH       = "DDICorpus2013.csv"
TFIDF_DIM      = 100
HIDDEN_DIM     = 64
GAT_OUT_DIM    = 128
GAT_HEADS      = 4
BATCH_SIZE     = 256
LR             = 1e-3
NUM_EPOCHS     = 20
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# 2. Load & Clean DataFrame
# ----------------------------
df = (
    pd.read_csv(CSV_PATH)           # read CSV
      .drop_duplicates()            # remove duplicate rows
      .reset_index(drop=True)       # reset index
)

def clean_drug(s):
    """Lowercase, strip, and remove non-alphanumerics from strings."""
    return re.sub(r'[^a-z0-9 ]', '', str(s).lower().strip())

# Apply cleaning to drug name columns
df['Drug_1'] = df['Drug_1_Name'].apply(clean_drug)
df['Drug_2'] = df['Drug_2_Name'].apply(clean_drug)
# Fill missing sentences with empty string
df['Sentence_Text'] = df['Sentence_Text'].fillna("")


# ----------------------------
# 3. Build Drug ↔ Index Map & Graph
# ----------------------------
# Unique drug list and index mapping
allDrugsData    = sorted(set(df['Drug_1']) | set(df['Drug_2']))
drugToIdxData   = {d: i for i, d in enumerate(allDrugsData)}
numNodesVal     = len(drugToIdxData)

# One-hot node features
x = torch.eye(numNodesVal, dtype=torch.float)

# Build bidirectional edge list for each observed pair
edgeList = []
for d1, d2 in zip(df['Drug_1'], df['Drug_2']):
    i1, i2 = drugToIdxData[d1], drugToIdxData[d2]
    edgeList += [[i1, i2], [i2, i1]]

# Convert edges to PyG format
edgeIndex = torch.tensor(edgeList, dtype=torch.long).t().contiguous()
data = Data(x=x, edge_index=edgeIndex).to(DEVICE)


# ----------------------------
# 4. Text Feature Extraction
# ----------------------------
tfidf = TfidfVectorizer(max_features=TFIDF_DIM)
# Fit + transform sentences → dense array
textFeatsData = tfidf.fit_transform(df['Sentence_Text']).toarray()


# ----------------------------
# 5. Positive/Negative Sampling
# ----------------------------
# Positive examples: (node1, node2, label=1, text_vector)
positive = [
    (drugToIdxData[r['Drug_1']],
     drugToIdxData[r['Drug_2']],
     1,
     textFeatsData[i])
    for i, r in df.iterrows()
]

# All possible undirected node pairs
allPairsVal = {(i, j) for i in range(numNodesVal)
                     for j in range(i+1, numNodesVal)}
posPairsVal = {(i, j) for i, j, _, _ in positive}

# Randomly sample equal number of negatives with zero text
negCands = list(allPairsVal - posPairsVal)
random.seed(42)
negative = [
    (i, j, 0, [0]*TFIDF_DIM)
    for i, j in random.sample(negCands, len(positive))
]

# Combine and split into train/val
edges       = positive + negative
trainEdges, valEdges = train_test_split(edges, test_size=0.2, random_state=42)


# ----------------------------
# 6. Dataset & DataLoaders
# ----------------------------
class EdgeDataset(Dataset):
    def __init__(self, edges):
        self.edges = edges

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        i1, i2, lbl, txt = self.edges[idx]
        return {
            "i1":   torch.tensor(i1, dtype=torch.long),
            "i2":   torch.tensor(i2, dtype=torch.long),
            "text": torch.tensor(txt, dtype=torch.float),
            "label":torch.tensor(lbl, dtype=torch.float)
        }

# DataLoaders for batching
trainLoader = DataLoader(EdgeDataset(trainEdges), batch_size=BATCH_SIZE, shuffle=True)
valLoader   = DataLoader(EdgeDataset(valEdges),   batch_size=BATCH_SIZE)


# ----------------------------
# 7. GAT & Classifier Models
# ----------------------------
class GATNet(torch.nn.Module):
    def __init__(self, in_c, out_c, heads=4):
        super().__init__()
        # First GAT layer: concat multiple heads
        self.conv1 = GATConv(in_c, out_c, heads=heads, concat=True)
        # Second GAT layer: single head, no concat
        self.conv2 = GATConv(out_c*heads, out_c, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class EdgeClassifierWithText(torch.nn.Module):
    def __init__(self, emb_dim, txt_dim, hidden_dim=64):
        super().__init__()
        # MLP on [emb1, emb2, text] → hidden → sigmoid
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2*emb_dim + txt_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, e1, e2, t):
        return self.net(torch.cat([e1, e2, t], dim=1))

# Instantiate models, optimizer, and loss
gat     = GATNet(in_c=numNodesVal, out_c=GAT_OUT_DIM, heads=GAT_HEADS).to(DEVICE)
edgeClf = EdgeClassifierWithText(emb_dim=GAT_OUT_DIM, txt_dim=TFIDF_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.Adam(
    list(gat.parameters()) + list(edgeClf.parameters()), lr=LR
)
criterion = torch.nn.BCELoss()


# ----------------------------
# 8. Training & Validation Loop
# ----------------------------
train_losses, val_losses = [], []
train_accs,    val_accs  = [], []
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    # --- Training ---
    gat.train(); edgeClf.train()
    epoch_loss = correct = total = 0

    for batch in trainLoader:
        optimizer.zero_grad()
        # compute fresh node embeddings each batch
        nodeEmb = gat(data.x, data.edge_index)

        emb1 = nodeEmb[batch["i1"]]
        emb2 = nodeEmb[batch["i2"]]
        text_feat = batch["text"].to(DEVICE)
        labels    = batch["label"].to(DEVICE).unsqueeze(1)

        preds = edgeClf(emb1, emb2, text_feat)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(labels)
        predicted = (preds >= 0.5).float()
        correct   += (predicted == labels).sum().item()
        total     += labels.size(0)

    train_loss = epoch_loss / total
    train_acc  = correct / total

    # --- Validation ---
    gat.eval(); edgeClf.eval()
    val_loss = correct = total = 0

    with torch.no_grad():
        nodeEmb = gat(data.x, data.edge_index)
        for batch in valLoader:
            emb1 = nodeEmb[batch["i1"]]
            emb2 = nodeEmb[batch["i2"]]
            text_feat = batch["text"].to(DEVICE)
            labels    = batch["label"].to(DEVICE).unsqueeze(1)

            preds = edgeClf(emb1, emb2, text_feat)
            val_loss += criterion(preds, labels).item() * len(labels)

            predicted = (preds >= 0.5).float()
            correct   += (predicted == labels).sum().item()
            total     += labels.size(0)

    val_loss /= total
    val_acc   = correct / total

    # Record metrics and save best model
    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc);     val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(gat.state_dict(),      "gat_best.pth")
        torch.save(edgeClf.state_dict(), "edge_best.pth")

    print(f"Epoch {epoch:02d}  "
          f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}  |  "
          f"Val loss:   {val_loss:.4f}, acc: {val_acc:.4f}")

print(f"\nBest validation accuracy: {best_val_acc:.4f}")


# ----------------------------
# 9. Plotting Metrics
# ----------------------------
import matplotlib.pyplot as plt

epochs = range(1, NUM_EPOCHS + 1)

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses,   label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.title("Loss Curve"); plt.show()

plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs,   label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.title("Accuracy Curve"); plt.show()

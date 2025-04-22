# ----------------------------
# 1. Imports and Dependencies
# ----------------------------
import re                    # for regex-based text cleaning
import random                # for selecting random templates
import torch                 # core PyTorch library
import torch.nn.functional as F  # activation functions (e.g., ELU)
import pandas as pd          # data loading and manipulation
from itertools import combinations  # generate all drug pairs
from sklearn.feature_extraction.text import TfidfVectorizer  # text feature extraction
from torch_geometric.data import Data      # graph data structure
from torch_geometric.nn import GATConv     # Graph Attention Network layer
from flask import Flask, request, jsonify, send_from_directory  # web API framework

# ----------------------------
# 2. Load and Clean Dataset
# ----------------------------
# Read the drug–drug interaction corpus, drop duplicates, reset index
df = (
    pd.read_csv("DDICorpus2013.csv")
      .drop_duplicates()
      .reset_index(drop=True)
)

def clean(s):
    """
    Clean text by lowercasing, stripping whitespace, and removing non-alphanumerics.
    """
    return re.sub(r'[^a-z0-9 ]', '', str(s).lower().strip())

# Apply cleaning to the drug name columns
df['Drug_1'] = df['Drug_1_Name'].apply(clean)
df['Drug_2'] = df['Drug_2_Name'].apply(clean)
# Ensure sentences are strings (fill NaN with empty string)
df['Sentence_Text'] = df['Sentence_Text'].fillna("")

# Build a mapping from drug name → node index in graph
all_drugs  = sorted(set(df['Drug_1']) | set(df['Drug_2']))
drug_to_idx = {d: i for i, d in enumerate(all_drugs)}

# ----------------------------
# 3. Text Feature Extraction
# ----------------------------
TFIDF_DIM  = 100  # limit TF‑IDF feature vector size
tfidf      = TfidfVectorizer(max_features=TFIDF_DIM)
# Fit TF‑IDF on all sentences, convert to dense array
text_feats = tfidf.fit_transform(df['Sentence_Text']).toarray()

# ----------------------------
# 4. Graph Construction
# ----------------------------
num_nodes = len(drug_to_idx)
# One-hot identity matrix as initial node features
x = torch.eye(num_nodes)
edges = []
# For each observed drug pair, add bidirectional edges
for d1, d2 in zip(df['Drug_1'], df['Drug_2']):
    i1, i2 = drug_to_idx[d1], drug_to_idx[d2]
    edges += [[i1, i2], [i2, i1]]
# Convert edge list to PyG’s edge_index format
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
data = Data(x=x, edge_index=edge_index)

# ----------------------------
# 5. GAT Model Definition
# ----------------------------
class GATNet(torch.nn.Module):
    def __init__(self, in_c, out_c, heads=4):
        super().__init__()
        # First GAT layer: in_c → out_c per head, concatenate head outputs
        self.conv1 = GATConv(in_c, out_c, heads=heads, concat=True)
        # Second GAT layer: (out_c * heads) → out_c, with single head, no concat
        self.conv2 = GATConv(out_c * heads, out_c, heads=1, concat=False)

    def forward(self, x, edge_index):
        # Apply ELU after first attention layer
        x = F.elu(self.conv1(x, edge_index))
        # Output final embeddings
        return self.conv2(x, edge_index)

# ----------------------------
# 6. Edge Classifier Definition
# ----------------------------
class EdgeClassifierWithText(torch.nn.Module):
    def __init__(self, emb_dim, txt_dim, hidden=64):
        super().__init__()
        # Simple MLP: [node1_emb, node2_emb, text_feats] → hidden → 1 → sigmoid
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2*emb_dim + txt_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, e1, e2, t):
        # Concatenate embeddings and text vector, then predict probability
        return self.net(torch.cat([e1, e2, t], dim=1))

# ----------------------------
# 7. Load Pretrained Models
# ----------------------------
DEVICE = torch.device("cpu")  # or 'cuda' if GPU is available

# Instantiate GAT and classifier; move to device
gat  = GATNet(num_nodes, 128, heads=4).to(DEVICE)
clf  = EdgeClassifierWithText(128, TFIDF_DIM, hidden=64).to(DEVICE)

# Load saved weights
gat.load_state_dict(torch.load("gat_best.pth", map_location=DEVICE))
clf.load_state_dict(torch.load("edge_best.pth", map_location=DEVICE))
gat.eval(); clf.eval()  # set both to evaluation mode

# Precompute node embeddings once (no grad needed)
with torch.no_grad():
    node_emb = gat(data.x, data.edge_index)

# ----------------------------
# 8. Interaction Thresholds & Templates
# ----------------------------
# Probability thresholds for high, medium, low interaction risk
HIGH_THR, MID_THR = 0.7, 0.4

# Example sentence templates for each risk level
HIGH_TEMPLATES = [
    "{d1} and {d2} have a high interaction possibility ({prob:.1f}%), which may lead to serious adverse effects.",
    "High risk: {d1}–{d2} interaction is {prob:.1f}% likely; adjust treatment accordingly.",
    "Alert: {d1} plus {d2} show a {prob:.1f}% chance of dangerous interactions.",
    "Caution! The combination of {d1} and {d2} has a {prob:.1f}% interaction probability.",
    "Warning: {d1} & {d2} interaction risk is {prob:.1f}%; monitor patient closely.",
    "{d1}-{d2} interaction is high ({prob:.1f}%), consider alternative therapy.",
    "High likelihood ({prob:.1f}%) of interaction between {d1} and {d2}.",
    "Strong interaction risk ({prob:.1f}%) detected for {d1} with {d2}.",
    "Serious: {d1} and {d2} may interact ({prob:.1f}% probability).",
    "Be careful: {prob:.1f}% chance of adverse effects when combining {d1} & {d2}."
    
]
MEDIUM_TEMPLATES = [
    "Medium interaction risk: {d1} & {d2} at {prob:.1f}%.",
    "Moderate ({prob:.1f}%) chance of interaction between {d1} and {d2}.",
    "{d1} plus {d2} have a {prob:.1f}% interaction likelihood; observe for side‑effects.",
    "Moderate risk ({prob:.1f}%) of {d1}–{d2} interaction; adjust if needed.",
    "Note: {prob:.1f}% chance that {d1} interacts with {d2}.",
    "Caution: medium ({prob:.1f}%) interaction probability for {d1} & {d2}.",
    "{d1} and {d2} show a medium interaction chance ({prob:.1f}%).",
    "Monitor: {d1}–{d2} interaction likelihood is {prob:.1f}%.",
    "Medium risk alert ({prob:.1f}%) for combining {d1} with {d2}.",
    "Watch closely: {d1} & {d2} have {prob:.1f}% interaction probability."
    
]
LOW_TEMPLATES = [
    "Low interaction risk: {d1} and {d2} at {prob:.1f}%.",
    "{d1} + {d2} unlikely to interact ({prob:.1f}% chance).",
    "{d1} and {d2} show a minimal ({prob:.1f}%) interaction likelihood.",
    "Low-risk combination: {d1}–{d2} at {prob:.1f}%.",
    "Little chance ({prob:.1f}%) of interaction between {d1} & {d2}.",
    "{d1} with {d2} has low ({prob:.1f}%) interaction probability.",
    "Safe: only {prob:.1f}% chance of {d1}–{d2} interaction.",
    "Minor ({prob:.1f}%) interaction risk for {d1} and {d2}.",
    "Low concern: {d1} & {d2} probability is {prob:.1f}%.",
    "Good news: {d1} plus {d2} have a {prob:.1f}% low interaction risk."
   
]

# ----------------------------
# 9. Prediction Function
# ----------------------------
def predict_interactions(drugs):
    """
    Given 2–4 drug names, predict pairwise interaction probabilities
    and return human‑readable sentences.
    """
    if not (2 <= len(drugs) <= 4):
        raise ValueError("Enter between 2 and 4 drug names.")
    cleaned = [clean(d) for d in drugs]
    unique  = sorted(set(cleaned))
    out = []
    # For each unique pair
    for d1, d2 in combinations(unique, 2):
        i1, i2 = drug_to_idx.get(d1), drug_to_idx.get(d2)
        if i1 is None or i2 is None:
            continue  # skip unknown drug names
        # Extract node embeddings for both drugs
        e1, e2 = node_emb[i1].unsqueeze(0), node_emb[i2].unsqueeze(0)
        # Find a representative TF‑IDF vector for this pair, or zeros if none
        mask = (df['Drug_1']==d1) & (df['Drug_2']==d2)
        if mask.any():
            # use the first matching sentence’s features
            txt = torch.tensor(text_feats[mask.idxmax()], dtype=torch.float).unsqueeze(0)
        else:
            txt = torch.zeros(1, TFIDF_DIM)
        # Predict interaction probability
        p = clf(e1, e2, txt).item()
        # Choose appropriate template set by threshold
        tpl = (HIGH_TEMPLATES if p >= HIGH_THR
               else MEDIUM_TEMPLATES if p >= MID_THR
               else LOW_TEMPLATES)
        # Format and append a random template
        out.append(random.choice(tpl).format(d1=d1, d2=d2, prob=p*100))
    return out

# ----------------------------
# 10. Flask API Setup
# ----------------------------
app = Flask(__name__, static_folder='')

@app.route('/')
def index():
    # Serve the main HTML page
    return send_from_directory('', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    # Receive JSON payload with {'drugs': [...]}
    js = request.get_json(force=True)
    try:
        res = predict_interactions(js.get('drugs', []))
    except ValueError as e:
        # Return a 400 error if input is invalid
        return jsonify({'error': str(e)}), 400
    # Return predictions as JSON
    return jsonify({'results': res})

if __name__ == '__main__':
    # Run Flask in debug mode on localhost:5000
    app.run(debug=True)


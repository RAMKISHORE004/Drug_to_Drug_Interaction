# app.py
import re
import random
import torch
import torch.nn.functional as F
import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from flask import Flask, request, jsonify, send_from_directory

# —————————————
# 1. Data + TF‑IDF
# —————————————
df = (
    pd.read_csv("DDICorpus2013.csv")
      .drop_duplicates()
      .reset_index(drop=True)
)
def clean(s):
    return re.sub(r'[^a-z0-9 ]', '', str(s).lower().strip())

df['Drug_1'] = df['Drug_1_Name'].apply(clean)
df['Drug_2'] = df['Drug_2_Name'].apply(clean)
df['Sentence_Text'] = df['Sentence_Text'].fillna("")

all_drugs  = sorted(set(df['Drug_1']) | set(df['Drug_2']))
drug_to_idx = {d: i for i, d in enumerate(all_drugs)}

TFIDF_DIM  = 100
tfidf      = TfidfVectorizer(max_features=TFIDF_DIM)
text_feats = tfidf.fit_transform(df['Sentence_Text']).toarray()

# —————————————
# 2. Build Graph
# —————————————
num_nodes = len(drug_to_idx)
x = torch.eye(num_nodes)
edges = []
for d1, d2 in zip(df['Drug_1'], df['Drug_2']):
    i1, i2 = drug_to_idx[d1], drug_to_idx[d2]
    edges += [[i1, i2], [i2, i1]]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
data = Data(x=x, edge_index=edge_index)

# —————————————
# 3. Model Definitions
# —————————————
class GATNet(torch.nn.Module):
    def __init__(self, in_c, out_c, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_c, out_c, heads=heads, concat=True)
        self.conv2 = GATConv(out_c * heads, out_c, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class EdgeClassifierWithText(torch.nn.Module):
    def __init__(self, emb_dim, txt_dim, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2*emb_dim + txt_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, e1, e2, t):
        return self.net(torch.cat([e1, e2, t], dim=1))

# —————————————
# 4. Load Weights & Precompute
# —————————————
DEVICE = torch.device("cpu")
gat  = GATNet(num_nodes, 128, heads=4).to(DEVICE)
clf  = EdgeClassifierWithText(128, TFIDF_DIM, hidden=64).to(DEVICE)
gat.load_state_dict(torch.load("gat_best.pth", map_location=DEVICE))
clf.load_state_dict(torch.load("edge_best.pth", map_location=DEVICE))
gat.eval(); clf.eval()

with torch.no_grad():
    node_emb = gat(data.x, data.edge_index)

# —————————————
# 5. Templates & Thresholds
# —————————————
HIGH_THR, MID_THR = 0.7, 0.4

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

# —————————————
# 6. Inference fn
# —————————————
def predict_interactions(drugs):
    if not (2 <= len(drugs) <= 4):
        raise ValueError("Enter between 2 and 4 drug names.")
    cleaned = [clean(d) for d in drugs]
    unique  = sorted(set(cleaned))
    out = []
    for d1, d2 in combinations(unique, 2):
        i1, i2 = drug_to_idx.get(d1), drug_to_idx.get(d2)
        if i1 is None or i2 is None: continue
        e1, e2 = node_emb[i1].unsqueeze(0), node_emb[i2].unsqueeze(0)
        mask = (df['Drug_1']==d1)&(df['Drug_2']==d2)
        if mask.any():
            txt = torch.tensor(text_feats[mask.idxmax()],dtype=torch.float).unsqueeze(0)
        else:
            txt = torch.zeros(1, TFIDF_DIM)
        p = clf(e1, e2, txt).item()
        tpl = (HIGH_TEMPLATES if p>=HIGH_THR
               else MEDIUM_TEMPLATES if p>=MID_THR
               else LOW_TEMPLATES)
        out.append(random.choice(tpl).format(d1=d1, d2=d2, prob=p*100))
    return out

# —————————————
# 7. Flask Server
# —————————————
app = Flask(__name__, static_folder='')
@app.route('/')
def index():       return send_from_directory('', 'index.html')
@app.route('/predict', methods=['POST'])
def predict_api():
    js = request.get_json(force=True)
    try:
        res = predict_interactions(js.get('drugs', []))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'results': res})

if __name__ == '__main__':
    app.run(debug=True)

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# Step 1: Prepare Graph Data
le = LabelEncoder()
df_balanced_improved['Drug_1_ID'] = le.fit_transform(df_balanced_improved['Drug_1_Name'])
df_balanced_improved['Drug_2_ID'] = le.transform(df_balanced_improved['Drug_2_Name'])

# Convert to Edge List (Graph Construction)
edges = torch.tensor(
    list(zip(df_balanced_improved['Drug_1_ID'], df_balanced_improved['Drug_2_ID'])), dtype=torch.long
).t().contiguous()

# Node Features: Use TF-IDF Embeddings
num_drugs = df_balanced_improved[['Drug_1_ID', 'Drug_2_ID']].max().max() + 1
node_features = torch.zeros((num_drugs, 128))  # Placeholder for embeddings

# Labels
labels = torch.tensor(df_balanced_improved['Is_DDI'].values, dtype=torch.long)

data = Data(x=node_features, edge_index=edges, y=labels)

# Step 2: Define Graph Attention Network (GAT)
class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)
    
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Step 3: Train & Evaluate
train_data, test_data = train_test_split([data], test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATModel(in_channels=128, hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Training Loop
def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.y], data.y)
        loss.backward()
        optimizer.step()

def test():
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(test_loader.dataset)

for epoch in range(50):
    train()
    acc = test()
    print(f'Epoch {epoch+1}, Accuracy: {acc:.4f}')


"""
To integrate Graph Attention Networks (GAT) for Drug-to-Drug Interaction (DDI) prediction, we need to:

Construct a graph representation where nodes represent drugs, and edges represent interactions.

Utilize the Deep Graph Library (DGL) or PyTorch Geometric (PyG) for graph-based learning.
\
Implement GAT for DDI prediction.

Steps:

Create a Drug Interaction Graph: Convert the dataset into a graph where drugs are nodes, and interactions form edges.

Prepare Node Features: Use TF-IDF or BERT embeddings as node features.

Train a GAT Model: Utilize PyTorch Geometric (PyG) or DGL.

Adding Graph Attention Networks (GAT) Code

I'll modify your pipeline to include a GAT-based model using PyTorch Geometric.

You'll need to install PyTorch Geometric:

pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
"""

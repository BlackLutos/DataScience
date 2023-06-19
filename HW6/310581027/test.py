import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sklearn.metrics import roc_auc_score
import pandas as pd
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.feature, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(num_features=10, num_classes=2).to(device)
model.load_state_dict(torch.load('best_model.pth'))

# Load the test data, and convert the data to float type
test_data = torch.load('test_sub-graph_tensor_noLabel.pt').to(device)
test_data.feature = test_data.feature.float() # Convert feature data to float type
test_mask = torch.from_numpy(np.load('test_mask.npy')).to(device)

print('Start testing...')

model.eval()
with torch.no_grad():
    predictions = model(test_data)

# Get the probabilities corresponding to class 1, which I assume are your 'anomaly scores'
probabilities = predictions[:, 1].cpu().numpy() # Added .cpu()

# Filter the probabilities to include only those corresponding to the nodes specified by your test_mask
filtered_probabilities = probabilities[test_mask.cpu().numpy()] # Ensure mask is on CPU and converted to numpy

# Get the indices of the nodes specified by your test_mask
node_indices = np.arange(len(test_mask))[test_mask.cpu().numpy()] # Ensure mask is on CPU and converted to numpy

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame({
    'node idx': node_indices,
    'node anomaly score': filtered_probabilities
})
df.to_csv('310581027.csv', index=False)

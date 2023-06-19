import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sklearn.metrics import roc_auc_score
import pandas as pd

train_data = torch.load('train_sub-graph_tensor.pt')
train_mask = np.load('train_mask.npy')

# Convert NumPy array to PyTorch tensor
train_mask = torch.from_numpy(train_mask)

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
# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, 16)
#         self.conv2 = GCNConv(num_features+16, num_classes)  # Change the input size to the sum of original feature size and the size of aggregate feature

#     def forward(self, data):
#         x, edge_index = data.feature, data.edge_index

#         x1 = self.conv1(x, edge_index)
#         x1 = F.relu(x1)
#         x1 = F.dropout(x1, training=self.training)

#         # concatenate the original node feature and the aggregated feature
#         x2 = torch.cat([x, x1], dim=1)  

#         x2 = self.conv2(x2, edge_index)

#         return F.log_softmax(x2, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(num_features=10, num_classes=2).to(device)
data = train_data.to(device)
data.feature = data.feature.float() # Convert feature data to float type
data.label = data.label.float() # Convert label data to float type

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Initialize the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

# First, create an empty label array
full_label = np.empty(39357, dtype=np.float32)  
# Set all values to a special value (for example, -1), this value does not appear in your actual labels
full_label.fill(-1)  
# Fill in the labels of the training nodes
full_label[train_mask] = train_data.label.cpu().numpy()

# Then convert this label array into a PyTorch tensor, and set it as the label of data
data.label = torch.from_numpy(full_label).long().to(device)

model.train()
loss_all = 1
print('Start training...')
for epoch in range(100000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[train_mask], data.label[train_mask])
    loss.backward()
    optimizer.step()

    # # Step the learning rate scheduler
    # scheduler.step()

    # Save the model with the smallest loss
    if loss.item() < loss_all:
        loss_all = loss.item()
        print('Epoch {}, Loss: {}'.format(epoch, loss.item()))
        torch.save(model.state_dict(), 'best_model.pth')

# Load the test data, and convert the data to float type
test_data = torch.load('test_sub-graph_tensor_noLabel.pt').to(device)
test_data.feature = test_data.feature.float() # Convert feature data to float type
test_mask = torch.from_numpy(np.load('test_mask.npy')).to(device)

model.load_state_dict(torch.load('best_model.pth'))

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




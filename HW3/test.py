import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms
from torchvision.models import resnet18
import torchvision.models as models
import pickle
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on {device}")

### Load data

class FewShotTestDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.sup_images = torch.from_numpy(data_dict['sup_images']).float()
        self.sup_labels = torch.from_numpy(data_dict['sup_labels']).long()
        self.qry_images = torch.from_numpy(data_dict['qry_images']).float()
    
    def get_labels(self):
        return self.sup_labels
    
test_data = FewShotTestDataset('test.pkl')

### Load model

class FewShotModel(nn.Module):
    def __init__(self):
        super(FewShotModel, self).__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 64)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet18.fc(x)
        return x

model = FewShotModel().to(device)
model.load_state_dict(torch.load('model.pth'))

### Test

def test(model, test_data, num_task=600):
    sup_images = test_data.sup_images
    sup_labels = test_data.sup_labels
    qry_images = test_data.qry_images
    num_correct = 0
    pred_arr = []
    for i in range(num_task):
        support_set = sup_images[i].to(device)
        support_labels = sup_labels[i].to(device)
        query_set = qry_images[i].to(device)
        query_labels = np.arange(5)
        query_labels = np.tile(query_labels, 5)
        query_labels = torch.from_numpy(query_labels).long().to(device)
        model.eval()
        with torch.no_grad():
            support_features = model(support_set)
            query_features = model(query_set)
            dists = torch.cdist(query_features, support_features)
            _, preds = torch.min(dists, dim=1)
            preds = preds.cpu().numpy()
            
            label_dict = {}
            for j in range(len(support_labels)):
                label_dict[preds[j]] = support_labels[preds[j]]
            
            for j in range(len(preds)):
                preds[j] = label_dict[preds[j]]

            pred_arr.append(preds)
    
    pred_arr = np.array(pred_arr)
    pred_arr = pred_arr.reshape(-1)
    pred_data = {"Category":pred_arr}
    df_pred = pd.DataFrame(pred_data)
    df_pred.to_csv('310581027_pred.csv', index_label='Id')

test(model, test_data)

print('Testing done')
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import os
import argparse
import shutil

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        x = self.resnet50(x)
        return x
    
parser = argparse.ArgumentParser()
parser.add_argument("arg1")
args = parser.parse_args()
read_file = str(args.arg1)
target_folder = 'predict_data\pred'
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

with open(read_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        #print(line)
        filename = os.path.basename(line)
        source_folder = os.path.dirname(line)
        #print(source_folder)
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        if source_folder:
            #print(source_folder)
            shutil.copy(source_path, target_path)
            #print(target_path)

    
    
model = BinaryClassifier()
model.load_state_dict(torch.load('model2.pth'))
model.eval()


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


predict_data = datasets.ImageFolder('predict_data', transform=data_transforms)


predict_loader = DataLoader(predict_data, batch_size=32)


predictions = []
with torch.no_grad():
    for inputs, labels in predict_loader:
        outputs = model(inputs)
        predictions += list(torch.sigmoid(outputs).squeeze().cpu().numpy())

acc = []
for i,pred in enumerate(predictions):
    if pred > 0.5:
        print(f"{predict_data.imgs[i][0]} is popular.")
        acc.append(1)
    else:
        print(f"{predict_data.imgs[i][0]} is not popular.")
        acc.append(0)
print(acc)
with open('310581027.txt', 'w') as f:
    for i in acc:
        f.write(str(i))
shutil.rmtree(target_folder)

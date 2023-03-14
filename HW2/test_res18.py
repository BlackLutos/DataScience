import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torchvision.models as models
import torch.nn.utils.prune as prune
import pandas as pd
from torchinfo import summary
# from torchsummary import summary

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def prune_weights(model):
    parameters_to_prune =[]
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')

def test_resnet18_on_fashion_mnist(weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)

    # ResNet-18 Model class
    class StudentModel(nn.Module):
        def __init__(self):
            super(StudentModel, self).__init__()
            self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = self.resnet18.fc.in_features
            self.resnet18.fc = nn.Linear(num_ftrs, 10)

            # for name, module in self.resnet18.named_modules():
            #     if isinstance(module, nn.Conv2d):
            #         prune.l1_unstructured(module, name='weight', amount=0.2)

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
        def fuse_model(self):
            torch.quantization.fuse_modules(self.resnet18, ['conv1', 'bn1', 'relu'], inplace=True)    


    net = StudentModel().to(device)
    # net = torch.load(weights_path)

    checkpoint = torch.load(weights_path)
    # net.load_state_dict(checkpoint['model_state_dict'])
    net.load_state_dict(checkpoint,strict=False)
    
    net.eval()
    torch.quantization.quantize_dynamic(net, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False)
    # net.fuse_model()
    # prune_weights(net)
    # module = net.resnet18.conv1
    # prune.random_unstructured(module, name="weight", amount=0.3)
    
    # summary(net, (1, 3, 28, 28))
    # summary(net, (32, 3, 28, 28), depth=3)
    summary(net)

    correct = 0
    total = 0
    pred_arr = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_arr.append(predicted.item())

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the {total} test images: {accuracy:.2f} %")
    # total_numel = 0
    # for name, param in net.named_parameters():
    #     if 'weight' in name:
    #         total_numel += torch.sum(param != 0)
    # print("Total number of nonzero parameters after pruning: ", total_numel)

    pred_data = {"pred":pred_arr}
    df_pred = pd.DataFrame(pred_data)
    df_pred.to_csv('310581027_pred.csv', index_label='id')

    return accuracy


        
def main():
    ckpt_path ="./model.pth"
    test_resnet18_on_fashion_mnist(ckpt_path)
    print("----------------------------------------------")

if __name__ == "__main__":
    main()        
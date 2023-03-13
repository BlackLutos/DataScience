import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torchsummary import summary

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

        # Resnet-50
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet50.fc(x)
        return x

    # ResNet-18
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10)

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



# def loss_fn_kd(outputs, labels, teacher_outputs, params):
#     """
#     Compute the knowledge-distillation (KD) loss given outputs, labels.
#     "Hyperparameters": temperature and alpha
#     NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
#     and student expects the input tensor to be log probabilities! See Issue #2
#     """
#     alpha = params.alpha
#     T = params.temperature
#     KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
#                              F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
#                              F.cross_entropy(outputs, labels) * (1. - alpha)
 
#     return KD_loss

softmax_op = nn.Softmax(dim=1)
mseloss_fn = nn.MSELoss()
def loss_fn(scores, targets, temperature = 5):
    soft_targets = softmax_op(teacher_outputs / temp)
    soft_student = softmax_op(student_outputs / temp)
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(soft_student, dim=1), soft_targets)
    # soft_pred = softmax_op(scores / temperature)
    # soft_targets = softmax_op(targets / temperature)
    # loss = mseloss_fn(soft_pred, soft_targets)
    # return loss

checkpoint = torch.load("resnet-50.pth")
teacher_model = TeacherModel().to(device)
# teacher_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
teacher_model.load_state_dict(checkpoint['model_state_dict'])
teacher_model.eval()

# transform
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=3),  # gray to 3 channel
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

train_data = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    transform=transform,
    download=True
)

test_data = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=False,
    transform=transform,
    download=True
)

batch_size = 32

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


# teacher_model = teacher_model.to(device)
student_model = StudentModel().to(device)

# temperature = 3
# alpha = 0.5
# num_epochs = 10
# batch_size = 128
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=lr)

summary(student_model, (3, 28, 28))

num_epochs = 1

model_num = 1

for epoch in range(num_epochs):
    train_loss = 0.0
    
    for data in train_loader:
        # Get inputs and labels
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs = inputs.repeat(1, 3, 1, 1)
        optimizer.zero_grad()

        # Compute teacher model outputs
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
            # teacher_outputs = teacher_outputs.view(-1)
        # Compute student model outputs
        student_outputs = student_model(inputs)
        # Compute loss
        temp = 5
        loss = loss_fn(student_outputs, teacher_outputs, temp)

        # Backward pass and update weights
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = student_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network {model_num} times on the {total} test images: {accuracy:.2f} %")
    total_numel = 0
    for name, param in student_model.named_parameters():
        if 'weight' in name:
            total_numel += torch.sum(param != 0)
    print("Total number of nonzero parameters after pruning: ", total_numel)

    model_file_name = "./model/" + str(model_num) + ".pth"
    torch.save(student_model.state_dict(), model_file_name)
    model_num += 1
    




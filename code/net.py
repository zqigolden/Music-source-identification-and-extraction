import torch
from dataset import ClassificationDataset
from torchvision.models import resnet18, resnet50
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

def init_weights(m):
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.0001)

class ClassificationNet(nn.Module):
    def __init__(self, hidden_layer=256, num_classes=3):
        super(ClassificationNet, self).__init__()
        self.pointwise = nn.Conv2d(1, 3, 1)
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_layer)
        self.bn = nn.BatchNorm1d(hidden_layer)
        self.dropout = nn.Dropout(p=0.5)
        self.classification_fc = nn.Linear(hidden_layer, num_classes)
        init_weights(self.pointwise)
        init_weights(self.resnet.fc)
        init_weights(self.classification_fc)

    def forward(self, x):
        x = self.pointwise(x)
        x = self.resnet(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.classification_fc(x)
        x = torch.sigmoid(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.pointwise.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

if __name__ == '__main__':
    net = ClassificationNet()
    # print(net)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = ClassificationDataset(root='./datasets/MUSDB18', subsets=['train'], transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    print(net(next(iter(trainloader))[0]))
from cgi import test
from cmath import log
import logging
import os
import torch
from torch import nn
from net import ClassificationNet
import torchvision.transforms as transforms
import dataset
from pathlib import Path
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# parameters
batch_size = 32
epoches = 40
use_gpu = True
checkpoint_dir = 'checkpoint'
num_workers = 4

# init logger
FORMAT = '%(levelname)s %(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler('train.log'))
logger.info('Start, cwd=%s', os.getcwd())
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    # transforms.Normalize(mean=[4.0], std=[44.0])
])
trainset = dataset.ClassificationDataset(root='./datasets/musdb18-wav', subsets=['train'], transform=transform, is_wav=True, use_cache=True)
# trainset = dataset.PartialDataset(root='./datasets/musdb18-wav', subsets=['test'], transform=transform, is_wav=True, use_cache=True, end_at=100)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testset = dataset.ClassificationDataset(root='./datasets/musdb18-wav', subsets=['test'], transform=transform, is_wav=True, use_cache=True)
# testset = dataset.PartialDataset(root='./datasets/musdb18-wav', subsets=['test'], transform=transform, is_wav=True, use_cache=True, end_at=100)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
logger.debug('dataset loaded')
classes = ('DRUMS', 'BASS', 'VOCALS', 'OTHER')

net = ClassificationNet(num_classes=3)
if use_gpu:
    net = net.cuda()

def CrossEntropyLossPerClass(loss, output, target):
    loss = loss(output, target)
    loss = loss.mean(dim=0)
    return loss

criterion = nn.BCELoss() # for multi-label classification
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5) #add regularization term by weight decay
schedular = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # decay learning rate by a factor of 0.5 every 5 epochs

# net.freeze()

for epoch in range(epoches):
    running_loss = 0.0
    net.train()
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        logger.debug('load data done')
        inputs, labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        logger.debug('to gpu done')
        optimizer.zero_grad()
        outputs = net(inputs)
        logger.debug('forward done')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        logger.debug('backward done')
        running_loss += loss.item()
        if i % 100 == 99:
            logger.info('[epoch:%d, iter:%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    schedular.step()
    chk_path = Path(checkpoint_dir)
    chk_path.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), chk_path / f'./model_epoch{epoch:03d}.pth')
    # if epoch == 5:
    #     net.unfreeze()
    if epoch % 2 == 1 or True:
        net.eval()
        errors = np.array([0, 0, 0])
        with torch.no_grad():
            running_loss = 0.0
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                inputs, labels = data
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                error = np.where(outputs.cpu().numpy() > 0.5, 1, 0) ^ np.array(labels.cpu(), dtype=int)
                errors += error.sum(axis=0)
            logger.info('[epoch:%d] test loss: %.3f' % (epoch + 1, running_loss / len(testloader)))
            logger.info('[epoch:{}] test error rate per class: {}'.format(epoch + 1, errors / len(testset)))
            if running_loss / len(testloader) < 0.25:
                print('early stopping')
                break

logger.info('Finished Training')
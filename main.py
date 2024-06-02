import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import models.resnet as resnet


import torch.optim as optim
from kan import KAN


from trainer import Trainer

args_dict = {}

transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 16

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model1 = resnet.resnet50_v2(weights=torchvision.models.ResNet50_Weights, pretrained=False)

# Freezing the layers of the resnet model
for param in model1.parameters():
    param.requires_grad = False

model1.fc = nn.Linear(model1.fc.in_features, 100)  
model1.fc = nn.Linear(model1.fc.in_features, 10)


model2 = resnet.resnet50_v2(weights=torchvision.models.ResNet50_Weights, pretrained=False)

# Freezing the layers of the resnet model
for param in model2.parameters():
    param.requires_grad = False





loss_function = nn.CrossEntropyLoss()

lr = 1e-4
args_dict['epochs'] = 15
args_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

args_dict['dataset_name'] = 'CIFAR10'
args_dict['trainloader'] = trainloader
args_dict['testloader'] = testloader

args_dict['model_name'] = 'ResNet50_v2'
args_dict['model'] = model1.to(args_dict['device'])
args_dict['optimizer'] = optim.SGD(model1.parameters(), lr = lr, momentum=0.9)
args_dict['scheduler'] = optim.lr_scheduler.CyclicLR(args_dict['optimizer'], base_lr=0.00005, max_lr=0.00001, step_size_up=3, mode='exp_range', gamma=0.5)


args_dict['loss_function'] = loss_function


if os.path.exists('saved_models\\'):
    pass
else:
    os.makedirs('saved_models\\')

args_dict['weights_save_path'] = 'saved_models'
args_dict['record_save_path'] = 'saved_models\\initial_trainings.txt'


trainer_obj = Trainer(args_dict)

## calculate number of parameters
print('Model:', args_dict['model_name'], ' No. of Parameters:', trainer_obj.calculate_no_of_parameters(args_dict['model']))

#trainer_obj.train_models()
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchsummary import summary
import torch

from efficientkan import  KAN as efficientKAN
from fastkan import FastKAN as fastKAN

from trainer import Trainer




# CNN model for CIFAR-10 with KANLinear
class EfficientKAN(nn.Module):
    def __init__(self, num_classes, dataset_name):
        super(EfficientKAN, self).__init__()
        
        if dataset_name == 'CIFAR10':
            self.input_size = 3072
        elif dataset_name == 'MNIST':
            self.input_size = 784
            
        self.efficientKAN = efficientKAN([self.input_size, 256, num_classes])

    def forward(self, x):
        print(x.shape)
        x = x.view(-1, self.input_size)
        x = self.efficientKAN(x)
        return x
    
# CNN model for CIFAR-10 with KANLinear
class MLP(nn.Module):
    def __init__(self, num_classes, dataset_name):
        super(MLP, self).__init__()
        
        if dataset_name == 'CIFAR10':
            self.input_size = 3072
        elif dataset_name == 'MNIST':
            self.input_size = 784
            
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.SELU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.mlp(x)
        return x
    
    
# CNN model for CIFAR-10 with fastKAN
class FastKAN(nn.Module):
    def __init__(self,  num_classes, dataset_name):
        super(FastKAN, self).__init__()
        if dataset_name == 'CIFAR10':
            self.input_size = 3072
        elif dataset_name == 'MNIST':
            self.input_size = 784
            
        self.fastKAN = fastKAN([self.input_size, 256, num_classes])
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fastKAN(x)
        return x


if __name__ == '__main__':
    ## device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## batch size
    batch_size = 64
    
    ## code_version_flag
    isMNIST = True
    
    ## dataset transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    ## dataset and dataloader
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    ## MNIST dataset and dataloader

    transform = transforms.Compose([  
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset_MNIST = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset_MNIST = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader_MNIST = DataLoader(train_dataset_MNIST, batch_size=batch_size, shuffle=True)
    test_loader_MNIST = DataLoader(test_dataset_MNIST, batch_size=batch_size, shuffle=False)
    
    
    ## define models
    MLP_model_1 = MLP(num_classes=10, dataset_name='CIFAR10').to(device)
    EfficientKAN_model_1 = EfficientKAN(num_classes=10, dataset_name='CIFAR10').to(device)
    FastKAN_model_1 = FastKAN(num_classes=10, dataset_name='CIFAR10').to(device)
    
    MLP_model_2 = MLP(num_classes=10, dataset_name='MNIST').to(device)
    EfficientKAN_model_2 = EfficientKAN(num_classes=10, dataset_name='MNIST').to(device)
    FastKAN_model_2 = FastKAN(num_classes=10, dataset_name='MNIST').to(device)
    
    
    ## define optimizer
    MLP_optimizer_1 = optim.AdamW(MLP_model_1.parameters(), lr=1e-3, weight_decay=1e-4)
    EfficientKAN_optimizer_1 = optim.AdamW(EfficientKAN_model_1.parameters(), lr=1e-3, weight_decay=1e-4)
    FastKAN_optimizer_1 = optim.AdamW(FastKAN_model_1.parameters(), lr=1e-3, weight_decay=1e-4)
    
    MLP_optimizer_2 = optim.AdamW(MLP_model_2.parameters(), lr=1e-3, weight_decay=1e-4)
    EfficientKAN_optimizer_2 = optim.AdamW(EfficientKAN_model_2.parameters(), lr=1e-3, weight_decay=1e-4)
    FastKAN_optimizer_2 = optim.AdamW(FastKAN_model_2.parameters(), lr=1e-3, weight_decay=1e-4)
    
    
    ## define loss function
    criterion = nn.CrossEntropyLoss()
    
    ## training schedular
    schedular_MLP_1 =  optim.lr_scheduler.ExponentialLR(MLP_optimizer_1, gamma=0.8)
    schedular_EfficientKAN_1 = optim.lr_scheduler.ExponentialLR(EfficientKAN_optimizer_1, gamma=0.8)
    schedular_FastKAN_1 = optim.lr_scheduler.ExponentialLR(FastKAN_optimizer_1, gamma=0.8)
    
    schedular_MLP_2 =  optim.lr_scheduler.ExponentialLR(MLP_optimizer_2, gamma=0.8)
    schedular_EfficientKAN_2 = optim.lr_scheduler.ExponentialLR(EfficientKAN_optimizer_2, gamma=0.8)
    schedular_FastKAN_2 = optim.lr_scheduler.ExponentialLR(FastKAN_optimizer_2, gamma=0.8)

    
    file_path = 'saved_models\\KAN_vs_MLP.txt'
    
    if isMNIST:
        models = [ MLP_model_2, EfficientKAN_model_2, FastKAN_model_2]
        model_names = ['MLP', 'EfficientKAN', 'FastKAN']
        dataset_name = ['MNIST']
        train_dataset_loader = [train_loader_MNIST]
        test_dataset_loader = [test_loader_MNIST]
        optimizers = [MLP_optimizer_2, EfficientKAN_optimizer_2, FastKAN_optimizer_2]
        schedulars = [schedular_MLP_2, schedular_EfficientKAN_2, schedular_FastKAN_2]
    else:
        models = [MLP_model_1, EfficientKAN_model_1, FastKAN_model_1]
        model_names = ['MLP', 'EfficientKAN', 'FastKAN']
        dataset_name = ['CIFAR10']
        train_dataset_loader = [train_loader]
        test_dataset_loader = [test_loader]
        optimizers = [MLP_optimizer_1, EfficientKAN_optimizer_1, FastKAN_optimizer_1]
        schedulars = [schedular_MLP_1, schedular_EfficientKAN_1, schedular_FastKAN_1]
    
    
    epochs = 10
    
    args_dict = {}
    args_dict['num_models'] = len(models)
    args_dict['num_datasets'] = 1
    for index in range(args_dict['num_datasets']):
        args_dict[('dataset_name', index)] = dataset_name[index]
        args_dict[('trainloader', index)] = train_dataset_loader[index]
        args_dict[('testloader', index)] = test_dataset_loader[index]
    
    
    args_dict['record_save_path'] = file_path
    args_dict['epochs'] = epochs
    args_dict['device'] = device
    args_dict['loss_function'] = criterion
    args_dict['weights_save_path'] = 'saved_models'
    
    for m in range(args_dict['num_models']):
        args_dict[('model', m)] = models[m]
        args_dict[('model_name', m)] = model_names[m]
        args_dict[('optimizers', m)] = optimizers[m]
        args_dict[('schedulers', m)] = schedulars[m]
        
        
        
    trainer = Trainer(args_dict)
    trainer.train_models()
    
    
  
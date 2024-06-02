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
    def __init__(self, num_classes):
        super(EfficientKAN, self).__init__()
        self.efficientKAN = efficientKAN([3072, 256, num_classes])

    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.efficientKAN(x)
        return x
    
# CNN model for CIFAR-10 with KANLinear
class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3072, 256),
            nn.SELU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.mlp(x)
        return x
    
    
# CNN model for CIFAR-10 with fastKAN
class FastKAN(nn.Module):
    def __init__(self,  num_classes):
        super(FastKAN, self).__init__()
        self.fastKAN = fastKAN([3072, 256, num_classes])
        
    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.fastKAN(x)
        return x


if __name__ == '__main__':
    ## device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## batch size
    batch_size = 64
    
    ## dataset transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  
    ])
    
    ## dataset and dataloader
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    ## define models
    MLP_model = MLP(10).to(device)
    EfficientKAN_model = EfficientKAN(10).to(device)
    FastKAN_model = FastKAN(10).to(device)
    
    ## define optimizer
    MLP_optimizer = optim.AdamW(MLP_model.parameters(), lr=1e-3, weight_decay=1e-4)
    EfficientKAN_optimizer = optim.AdamW(EfficientKAN_model.parameters(), lr=1e-3, weight_decay=1e-4)
    FastKAN_optimizer = optim.AdamW(FastKAN_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    ## define loss function
    criterion = nn.CrossEntropyLoss()
    
    ## training schedular
    schedular_MLP = optim.lr_scheduler.MultiStepLR(MLP_optimizer, milestones=[50, 100, 150], gamma=0.1)
    schedular_EfficientKAN = optim.lr_scheduler.MultiStepLR(EfficientKAN_optimizer, milestones=[50, 100, 150], gamma=0.1)
    schedular_FastKAN = optim.lr_scheduler.MultiStepLR(FastKAN_optimizer, milestones=[50, 100, 150], gamma=0.1)
    

    
    file_path = 'saved_models\\KAN_vs_MLP.txt'
    
    models = [MLP_model, EfficientKAN_model, FastKAN_model]
    model_names = ['MLP', 'EfficientKAN', 'FastKAN']
    dataset_name = 'CIFAR10'
    
    epochs = 10
    
    args_dict = {}
    args_dict['num_models'] = len(models)
    args_dict['num_datasets'] = 1
    for index in range(args_dict['num_datasets']):
        args_dict[('dataset_name', index)] = dataset_name
        args_dict[('trainloader', index)] = train_loader
        args_dict[('testloader', index)] = test_loader
    
    
    args_dict['record_save_path'] = file_path
    args_dict['epochs'] = epochs
    args_dict['device'] = device
    args_dict['loss_function'] = criterion
    args_dict['weights_save_path'] = 'saved_models'
    
    for m in range(args_dict['num_models']):
        args_dict[('model', m)] = models[m]
        args_dict[('model_name', m)] = model_names[m]
        args_dict[('optimizers', m)] = eval(model_names[m]+'_optimizer')
        args_dict[('schedulers', m)] = eval('schedular_'+model_names[m])
        
        
        
    trainer = Trainer(args_dict)
    trainer.train_models()
    
    
  
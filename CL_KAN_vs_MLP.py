import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchsummary import summary
import torch

from efficientkan import  KAN as efficientKAN
from fastkan import FastKAN as fastKAN
from kan import KANLayer

from continual_learning_trainer import ContinualLearningTrainer
from utils import DivideDataset



# CNN model for CIFAR-10 with KANLinear
class EfficientKAN(nn.Module):
    def __init__(self, num_classes, dataset_name, init_method='xavier'):
        super(EfficientKAN, self).__init__()
        
        if dataset_name == 'CIFAR10':
            self.input_size = 3072
        elif dataset_name == 'MNIST':
            self.input_size = 784
            
        self.efficientKAN = efficientKAN([self.input_size, 256, num_classes])
        self.init_weights(init_method=init_method)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.efficientKAN(x)
        return x
    
    ## initialize the weights
    def init_weights(self, init_method='xavier'):
        init_method = {
            'xavier': nn.init.xavier_normal_,
            'kaiming': nn.init.kaiming_normal_,
            'normal': nn.init.normal_,
        }

            
        ## initialize the weights for the efficientKAN 
        for m in self.efficientKAN.modules():
            for name, param in m.named_parameters():
                if name == 'base_weight' or name == 'spline_weight':
                    init_method[init_method](param)
                
                    
                    
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


class KAN_original(nn.Module):
    def __init__(self,  num_classes, dataset_name, device='cuda'):
        super(KAN_original, self).__init__()
        if dataset_name == 'CIFAR10':
            self.input_size = 3072
        elif dataset_name == 'MNIST':
            self.input_size = 784
            
        self.first_layer = KANLayer(in_dim=self.input_size, out_dim=256, num=5, k=3, device=device)
        self.second_layer = KANLayer(in_dim=256, out_dim=num_classes, num=5, k=3, device=device)
        self.activation = nn.SELU()
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x, _, _, _ = self.first_layer(x)
        x = self.activation(x)
        x ,_, _, _ = self.second_layer(x)

        return x


if __name__ == '__main__':
    ## device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # define models
    MLP_model_1 = MLP(num_classes=10, dataset_name='CIFAR10').to(device)
    EfficientKAN_model_1 = EfficientKAN(num_classes=10, dataset_name='CIFAR10').to(device)
    #FastKAN_model_1 = FastKAN(num_classes=10, dataset_name='CIFAR10').to(device)
    #KAN_original_model_1 = KAN_original(num_classes=10, dataset_name='CIFAR10', device=device).to(device)
    
    MLP_model_2 = MLP(num_classes=10, dataset_name='MNIST').to(device)
    EfficientKAN_model_2 = EfficientKAN(num_classes=10, dataset_name='MNIST').to(device)
    #FastKAN_model_2 = FastKAN(num_classes=10, dataset_name='MNIST').to(device)
    #KAN_original_model_2 = KAN_original(num_classes=10, dataset_name='MNIST').to(device)
    
    
    
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
    
    train_MNIST_task_divider = DivideDataset(train_dataset_MNIST, 5, list(range(10)))
    train_MNIST_task_datasets, tasks_classes = train_MNIST_task_divider.get_the_datasets()
    
    
    num_of_task = 5
    
    train_task_dataset_loader = {}
    for task in range(num_of_task):
        train_task_dataset_loader[task] = DataLoader(train_MNIST_task_datasets[task], batch_size=batch_size, shuffle=True)
        
        
    test_MNIST_task_divider = DivideDataset(test_dataset_MNIST, 5, list(range(10)))
    test_MNIST_task_datasets, tasks_classes = test_MNIST_task_divider.get_the_datasets()
    
    test_task_dataset_loader = {}
    for task in range(num_of_task):
        test_task_dataset_loader[task] = DataLoader(test_MNIST_task_datasets[task], batch_size=batch_size, shuffle=False)
        
    
    # train_loader_MNIST = DataLoader(train_dataset_MNIST, batch_size=batch_size, shuffle=True)
    # test_loader_MNIST = DataLoader(test_dataset_MNIST, batch_size=batch_size, shuffle=False)
    
    
    
  
    
    
    ## define optimizer
    MLP_optimizer_1 = optim.AdamW(MLP_model_1.parameters(), lr=1e-3, weight_decay=1e-4)
    EfficientKAN_optimizer_1 = optim.AdamW(EfficientKAN_model_1.parameters(), lr=1e-3, weight_decay=1e-4)
    #FastKAN_optimizer_1 = optim.AdamW(FastKAN_model_1.parameters(), lr=1e-3, weight_decay=1e-4)
    #KAN_original_optimizer_1 = optim.AdamW(KAN_original_model_1.parameters(), lr=1e-3, weight_decay=1e-4)
    
    
    MLP_optimizer_2 = optim.AdamW(MLP_model_2.parameters(), lr=1e-3, weight_decay=1e-4)
    EfficientKAN_optimizer_2 = optim.AdamW(EfficientKAN_model_2.parameters(), lr=1e-3, weight_decay=1e-4)
    #FastKAN_optimizer_2 = optim.AdamW(FastKAN_model_2.parameters(), lr=1e-3, weight_decay=1e-4)
    #KAN_original_optimizer_2 = optim.AdamW(KAN_original_model_2.parameters(), lr=1e-3, weight_decay=1e-4)
    
    
    ## define loss function
    criterion = nn.CrossEntropyLoss()
    
    ## training schedular
    schedular_MLP_1 =  optim.lr_scheduler.ExponentialLR(MLP_optimizer_1, gamma=0.8)
    schedular_EfficientKAN_1 = optim.lr_scheduler.ExponentialLR(EfficientKAN_optimizer_1, gamma=0.8)
    #schedular_FastKAN_1 = optim.lr_scheduler.ExponentialLR(FastKAN_optimizer_1, gamma=0.8)
    #schedular_original_KAN_1 = optim.lr_scheduler.ExponentialLR(KAN_original_optimizer_1, gamma=0.8)
    
    schedular_MLP_2 =  optim.lr_scheduler.ExponentialLR(MLP_optimizer_2, gamma=0.8)
    schedular_EfficientKAN_2 = optim.lr_scheduler.ExponentialLR(EfficientKAN_optimizer_2, gamma=0.8)
    #schedular_FastKAN_2 = optim.lr_scheduler.ExponentialLR(FastKAN_optimizer_2, gamma=0.8)
    #schedular_original_KAN_2 = optim.lr_scheduler.ExponentialLR(KAN_original_optimizer_2, gamma=0.8)
    
    

    
    if isMNIST:
        models = [ MLP_model_2, EfficientKAN_model_2] #, FastKAN_model_2, KAN_original_model_2]
        model_names = ['MLP', 'EfficientKAN', 'FastKAN', 'KAN_original']
        dataset_name = ['MNIST']
        train_dataset_loader = [train_task_dataset_loader]
        test_dataset_loader = [test_task_dataset_loader]
        optimizers = [MLP_optimizer_2, EfficientKAN_optimizer_2] #, FastKAN_optimizer_2, KAN_original_optimizer_2]
        schedulars = [schedular_MLP_2, schedular_EfficientKAN_2] #, schedular_FastKAN_2, schedular_original_KAN_2]
    else:
        models = [MLP_model_1, EfficientKAN_model_1] #, FastKAN_model_1]
        model_names = ['MLP', 'EfficientKAN', 'FastKAN']
        dataset_name = ['CIFAR10']
        train_dataset_loader = [train_loader]
        test_dataset_loader = [test_loader]
        optimizers = [MLP_optimizer_1, EfficientKAN_optimizer_1] #, FastKAN_optimizer_1]
        schedulars = [schedular_MLP_1, schedular_EfficientKAN_1] #, schedular_FastKAN_1]
    
    

    epoch_ditribution = {}
    for task in range(num_of_task):
        if task == 0:
            epoch_ditribution[task] = 7
        else:
            epoch_ditribution[task] = 5 
    
    
    
    file_path = 'saved_models\\CL_KAN_vs_MLP.txt'
    file_path_cl_tasks = 'saved_models\\CL_KAN_vs_MLP_tasks.txt'
    
    
    args_dict = {}
    args_dict['num_models'] = len(models)
    args_dict['num_datasets'] = 1
    for index in range(args_dict['num_datasets']):
        args_dict[('dataset_name', index)] = dataset_name[index]
        args_dict[('trainloader', index)] = train_dataset_loader[index]
        args_dict[('testloader', index)] = test_dataset_loader[index]
    
    
    args_dict['record_save_path'] = file_path
    args_dict['record_save_path_cl_tasks'] = file_path_cl_tasks
    args_dict['epoch_distribution'] = epoch_ditribution
    args_dict['device'] = device
    args_dict['loss_function'] = criterion
    args_dict['weights_save_path'] = 'saved_models'
    args_dict['num_tasks'] = num_of_task
    
    for m in range(args_dict['num_models']):
        args_dict[('model', m)] = models[m]
        args_dict[('model_name', m)] = model_names[m]
        args_dict[('optimizers', m)] = optimizers[m]
        args_dict[('schedulers', m)] = schedulars[m]
        
        
    cl_trainer = ContinualLearningTrainer(args_dict)
    cl_trainer.train_models()
    
    
  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchsummary import summary
import torch

from efficientkan import  KAN as efficientKAN
from models.vision_transformer import VisionTransformer, PatchEmbed


from continual_learning_trainer import ContinualLearningTrainer
from utils import DivideDataset


if __name__ == '__main__':
    ## device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## batch size
    batch_size = 16
    
    ## code_version_flag
    isMNIST = True
    
    ## dataset transform
    transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    
    num_of_task = 10
    
    ## dataset and dataloader
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform)
    

    
    train_CIFAR_task_divider = DivideDataset(train_dataset, num_of_task, list(range(100)))
    train_CIFAR_task_datasets, tasks_classes = train_CIFAR_task_divider.get_the_datasets()
    
    
    
    train_task_dataset_loader = {}
    for task in range(num_of_task):
        train_task_dataset_loader[task] = DataLoader(train_CIFAR_task_datasets[task], batch_size=batch_size, shuffle=True)
        
        
    test_Cifar_task_divider = DivideDataset(test_dataset, num_of_task, list(range(100)))
    test_CIFAR_task_datasets, tasks_classes = test_Cifar_task_divider.get_the_datasets()
    
    test_task_dataset_loader = {}
    for task in range(num_of_task):
        test_task_dataset_loader[task] = DataLoader(test_CIFAR_task_datasets[task], batch_size=batch_size, shuffle=False)
        
    
    ## replay dataset and loader
    replay_dataset = {}
    no_of_replay_samples = 2000
    for current_task in range(1, num_of_task):
        no_of_sample_per_task = no_of_replay_samples//current_task
        for task in range(current_task):
            ## get random no_of_sample_per_task samples from task dataset
            indices = torch.randperm(len(train_CIFAR_task_datasets[task])).tolist()[:no_of_sample_per_task]
            replay_samples = torch.utils.data.Subset(train_CIFAR_task_datasets[task], indices)
            if current_task not in replay_dataset:
                replay_dataset[current_task] = [replay_samples]
            else:
                replay_dataset[current_task].append(replay_samples)

    
    ## concatenate all replay samples and create a dataset loader
    replay_dataset_loader = {}
    for task in range(1, num_of_task):
        replay_dataset[task] = torch.utils.data.ConcatDataset(replay_dataset[task])
        replay_dataset_loader[task] = DataLoader(replay_dataset[task], batch_size=batch_size, shuffle=True)
        
    
    ## define models
    Vit_MLP = VisionTransformer(img_size=224, patch_size=16, in_c=3, num_classes=100, embed_dim=256, depth=8, num_heads=8, mlp_ratio=4.0, 
                               drop_path_ratio=0., embed_layer=PatchEmbed, isKAN=False).to(device)
    
    Vit_KAN = VisionTransformer(img_size=224, patch_size=16, in_c=3, num_classes=100, embed_dim=256, depth=8, num_heads=8, mlp_ratio=4.0,
                                 embed_layer=PatchEmbed,  isKAN=True).to(device)
    
    
    
    ## define optimizer
    Vit_MLP_optimizer = optim.AdamW(Vit_MLP.parameters(), lr=1e-3)
    Vit_KAN_optimizer = optim.AdamW(Vit_KAN.parameters(), lr=1e-3)
    
    
    
    ## define loss function
    criterion = nn.CrossEntropyLoss()
    
    ## training schedular
    Vit_MLP_schedular =  optim.lr_scheduler.StepLR(Vit_MLP_optimizer, step_size=5, gamma=0.9)
    Vit_KAN_schedular = optim.lr_scheduler.StepLR(Vit_KAN_optimizer, step_size=5, gamma=0.9)
    

    models = [Vit_MLP, Vit_KAN]
    model_names = ['Vit_MLP', 'Vit_KAN']
    dataset_name = ['CIFAR100']
    train_dataset_loader = [train_task_dataset_loader]
    test_dataset_loader = [test_task_dataset_loader]
    replay_dataset_loaders = [replay_dataset_loader]
    optimizers = [Vit_MLP_optimizer, Vit_KAN_optimizer]
    schedulars = [Vit_MLP_schedular, Vit_KAN_schedular]
   
    

    epoch_ditribution = {}
    for task in range(num_of_task):
        if task == 0:
            epoch_ditribution[task] = 35
        else:
            epoch_ditribution[task] = 15
    
    
    
    file_path = 'saved_models\\CL_ViT_KAN_vs_MLP_with_replay.txt'
    file_path_cl_tasks = 'saved_models\\CL_ViT_KAN_vs_MLP_tasks_with_replay.txt'
    
    
    args_dict = {}
    args_dict['num_models'] = len(models)
    args_dict['num_datasets'] = 1
    for index in range(args_dict['num_datasets']):
        args_dict[('dataset_name', index)] = dataset_name[index]
        args_dict[('trainloader', index)] = train_dataset_loader[index]
        args_dict[('testloader', index)] = test_dataset_loader[index]
        args_dict[('replayloader', index)] = replay_dataset_loaders[index]
    
    
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
    
    
  
import os

import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms


__all__ = ['DivideDataset']


class DivideDataset():
    def __init__(self, dataset=None, num_tasks=None, classes=None, seed=0):
        self.dataset = dataset
        self.num_tasks = num_tasks
        self.classes = classes
        self.seed = seed
        self.num_classes = len(classes)
        self.classes_per_task = self.num_classes // self.num_tasks
        self.task_classes = self.distribute_classes()
        self.task_datasets = self.divide_dataset()
        
        
    def distribute_classes(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        task_classes = {}
        
        for i in range(self.num_tasks):
            ## pick randomly self.task_per_class using the self.seed from the classes set
            task_classes[i] = list(np.random.choice(self.classes, self.classes_per_task, replace=False))
            
            ## remove the classes that are already picked
            self.classes = list(set(self.classes) - set(task_classes[i]))
        
        return task_classes
    
    
    def divide_dataset(self):
        task_datasets = {}
        
        for task in range(self.num_tasks):
            task_datasets[task] = []
            
            for i in range(len(self.dataset)):
                if self.dataset[i][1] in self.task_classes[task]:
                    task_datasets[task].append(self.dataset[i])
                    
        return task_datasets
    
    def get_the_datasets(self):
        return self.task_datasets, self.task_classes
    
    
    def save_datasets(self, save_path):
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
            
        for task in range(self.num_tasks):
            task_save_path = os.path.join(save_path, 'task_' + str(task))
            
            if os.path.isdir(task_save_path) == False:
                os.mkdir(task_save_path)
                
            for i in range(len(self.task_datasets[task])):
                torch.save(self.task_datasets[task][i], os.path.join(task_save_path, 'data_' + str(i) + '.pth'))
    
    def load_datasets(self, load_path):
        task_datasets = {}
        
        for task in range(self.num_tasks):
            task_load_path = os.path.join(load_path, 'task_' + str(task))
            task_datasets[task] = []
            
            for i in range(len(os.listdir(task_load_path))):
                task_datasets[task].append(torch.load(os.path.join(task_load_path, 'data_' + str(i) + '.pth')))
                
        return task_datasets
    

# if __name__ == '__main__':
  
#     print('Testing DivideDataset class')
    
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    
    # train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    
    # classes = list(range(10))
    # num_tasks = 5
    
    # divide_dataset = DivideDataset(train_dataset, num_tasks, classes)
    
    # task_datasets, task_classes = divide_dataset.get_the_datasets()
    
    # print(task_classes)
    # print(len(task_datasets[0]))
    
    # print(len(task_datasets[1]))
    
    
    
    
    
    
    
    
    
        
       


    
    
    
    
    

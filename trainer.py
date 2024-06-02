import torch
import torchvision

from kan.LBFGS import LBFGS
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import os

from timer import Timer

class Trainer():
    def __init__(self, args_dict):
        self.args_dict = args_dict
        self.timer = Timer()
        
    def calc_metrics(self, pred, target):
        _, pred = torch.max(pred, dim = 1)
        correct = (pred == target).sum()
        
        return correct/target.size(0)
        
    def save_model(self, model_name, dataset_name, epoch, model):
        folder_name = 'weights-' + str(model_name) + '-' + str(dataset_name)
        save_dir = os.path.join(self.args_dict['weights_save_path'], folder_name)
        
        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)
            
        save_name = 'Epoch-' + str(epoch) + '.pth'
        torch.save(model.state_dict(), os.path.join(save_dir, save_name))
    
    def load_model(self, model_name, dataset_name, epoch, model):
        folder_name = 'weights-' + str(model_name) + '-' + str(dataset_name)
        save_dir = os.path.join(self.args_dict['weights_save_path'], folder_name)
        
        load_name = 'Epoch-' + str(epoch) + '.pth'
        model.load_state_dict(torch.load(os.path.join(save_dir, load_name)))
        
        return model

    def calculate_no_of_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train_single(self, model, optimizer, inputs, labels):
        
        inputs = inputs.to(self.args_dict['device'])
        labels = labels.to(self.args_dict['device'])
        
        input_features = self.args_dict['feature_extractor'](inputs)
        
        self.timer.start()
        outputs = model(input_features)
        loss = self.args_dict['loss_function'](outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.timer.stop()
        
        metrics = self.calc_metrics(outputs, labels)
        
        return loss.item(), metrics, self.timer.get_elapsed_time()
        
    def eval_single(self):
        raise NotImplementedError('Eval single is not implemented')
        
    def train_models(self):
        
        dataset_name = self.args_dict['dataset_name'] 
        model = self.args_dict['model']
        model_name = self.args_dict['model_name'] 
        optimizer = self.args_dict['optimizer']
        
        with open(self.args_dict['record_save_path'], 'a') as f:
            f.write(f'Model name: ----------- {model_name} --------------')
            f.write('\n')
            f.write(f'Dataset name: ----------- {dataset_name} --------------\n')
        
        model.train()
        self.args_dict['feature_extractor'].eval()
        for epoch in range(self.args_dict['epochs']):
            loss_array = []
            accuracy_array = []
            epoch_time = []
            
            for data in tqdm(self.args_dict['trainloader']):
                inputs, labels = data
                inputs = inputs.to(self.args_dict['device'])
                labels = labels.to(self.args_dict['device'])
                
                if self.args_dict['kan'] == True:
                    loss, metrics, timer = model.train_single_KAN(timer=self.timer, optimizer=optimizer, inputs=inputs, labels=labels, args_dict=self.args_dict)
                else:
                    loss, metrics, timer = self.train_single(model, optimizer, inputs, labels)
                
                epoch_time.append(timer)
                metrics = metrics.cpu().numpy()
                
                loss_array.append(loss)
                accuracy_array.append(metrics)
            
            self.args_dict['scheduler'].step()
                
            with open(self.args_dict['record_save_path'], 'a') as f:
                f.write(f'Epoch: {epoch}')
                f.write('\n')
                f.write(f'Train Loss: {np.mean(loss_array)}')
                f.write('\n')
                f.write(f'Train Accuracy: {np.mean(accuracy_array)}')
                f.write('\n')
                f.write(f'Epoch Time: {np.sum(epoch_time)}')
                f.write('\n')
                f.write("Average time per iteration: " + str(np.mean(epoch_time)))
                f.write('\n')
                f.write("-------------------------------------------------\n")
            
            self.evaluate(model, self.args_dict['testloader'])
            self.save_model(model_name, dataset_name, epoch, model)
    
    def evaluate(self, model, test_loader):
        loss_array = []
        accuracy_array = []
        eval_time = []
        
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(self.args_dict['device'])
            labels = labels.to(self.args_dict['device'])
            
            model.eval()
            self.args_dict['feature_extractor'].eval()
            with torch.no_grad():
                
                input_features = self.args_dict['feature_extractor'](inputs)
                
                self.timer.start()
                outputs = model(input_features)
                self.timer.stop()
            
            eval_time.append(self.timer.get_elapsed_time())
            
            loss = self.args_dict['loss_function'](outputs, labels)
            
            metrics = self.calc_metrics(outputs, labels)
            
            metrics = metrics.cpu().numpy()
            
            loss_array.append(loss.item())
            accuracy_array.append(metrics)
            
        with open(self.args_dict['record_save_path'], 'a') as f:
            f.write(f'Validation Loss: {np.mean(loss_array)}')
            f.write('\n')
            f.write(f'Validation Accuracy: {np.mean(accuracy_array)}')
            f.write('\n')
            f.write(f'Validation Time: {np.sum(eval_time)}')
            f.write('\n')
            f.write("Average time per iteration: " + str(np.mean(eval_time)))
            f.write('\n')
            f.write("-------------------------------------------------\n")
    
    
            
    
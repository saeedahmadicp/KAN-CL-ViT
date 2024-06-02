import torch

from torch import nn
import numpy as np
import os

from matplotlib import pyplot as plt

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
        
    def eval_single(self):
        raise NotImplementedError('Eval single is not implemented')
    
    
    
    def train(self, model, train_loader, optimizer, epoch):
    
        loss_array = []
        accuracy_array = []
        epoch_time = []
        timer = Timer()
        
        device = self.args_dict['device']
        file_path = self.args_dict['record_save_path']
        
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            timer.start()
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            timer.stop()
            
            loss_array.append(loss.item())
            accuracy = self.calc_metrics(output, target)
            accuracy_array.append(accuracy.item())
            epoch_time.append(timer.get_elapsed_time())
            
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
        print(f'Train Epoch: {epoch} Loss: {np.mean(loss_array)} Accuracy: {np.mean(accuracy_array)}')
        with open(file_path, 'a') as f:
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
            
        
        return np.mean(loss_array), np.mean(accuracy_array)
    
        
    def train_models(self):
        
        for m in range(self.args_dict['num_models']):
            dataset_index = m % self.args_dict['num_datasets']
            dataset_name = self.args_dict[('dataset_name', dataset_index)] 
            model = self.args_dict[('model', m)]
            model_name = self.args_dict[('model_name', m)] #model_inf[1]
            optimizer = self.args_dict[('optimizers', m)]
            scheduler = self.args_dict[('schedulers', m)]
            
            with open(self.args_dict['record_save_path'], 'a') as f:
                f.write(f'Model name: ----------- {model_name} --------------')
                f.write('\n')
                f.write(f'Dataset name: ----------- {dataset_name} --------------\n')
                
            
            print(f'Model name: ----------- {model_name} --------------')
            
            train_losses = []
            train_accuracies = []
            test_losses = []
            test_accuracies = []
            
            for epoch in range(self.args_dict['epochs']):
                train_loss, train_accuracy = self.train(model=model, train_loader=self.args_dict[('trainloader', dataset_index)], optimizer=optimizer, epoch=epoch)
                test_loss, test_accuracy = self.evaluate(model, self.args_dict[('testloader', dataset_index)], scheduler)
                self.save_model(model_name, dataset_name, epoch, model)
                
                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)
                test_accuracies.append(test_accuracy)
                test_losses.append(test_loss)
            
            self.plot_results(train_losses, train_accuracies, test_losses, test_accuracies, model_name, dataset_name)
                
    
    def plot_results(self, train_losses, train_accuracies, test_losses, test_accuracies, model_name, dataset_name):
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        
        save_dir = 'graphs'
        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)
            
        path = os.path.join(save_dir, f'Loss-{model_name}-{dataset_name}.png')
        plt.savefig(path)
        
        plt.figure()
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch')
        plt.legend()
        
        path = os.path.join(save_dir, f'Accuracy-{model_name}-{dataset_name}.png')
        plt.savefig(path)
        
        
    
    
    def evaluate(self, model, test_loader, scheduler):
        loss_array = []
        accuracy_array = []
        eval_time = []
        
        model.eval()
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(self.args_dict['device'])
            labels = labels.to(self.args_dict['device'])
            
            with torch.no_grad():
                self.timer.start()
                outputs = model(inputs)
                self.timer.stop()
            
            eval_time.append(self.timer.get_elapsed_time())
            
            loss = self.args_dict['loss_function'](outputs, labels)
            
            metrics = self.calc_metrics(outputs, labels)
            
            metrics = metrics.cpu().numpy()
            
            loss_array.append(loss.item())
            accuracy_array.append(metrics.item())
        
        
        if scheduler is not None:
            scheduler.step(np.mean(loss_array))
        
        print(f'Validation Loss: {np.mean(loss_array)}')
        print(f'Validation Accuracy: {np.mean(accuracy_array)}')
        print(f'Validation Time: {np.sum(eval_time)}')
            
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
            
            
        return np.mean(loss_array), np.mean(accuracy_array)
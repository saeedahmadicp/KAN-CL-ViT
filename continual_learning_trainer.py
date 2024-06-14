import torch

from torch import nn
import numpy as np
import os

from matplotlib import pyplot as plt

from timer import Timer


class ContinualLearningTrainer():
    def __init__(self, args_dict):
        self.args_dict = args_dict
        self.timer = Timer()
        self.num_tasks = args_dict['num_tasks']
        self.current_task = 0
        self.average_accuracy_after_task = []
        self.each_task_accuracy = []
        self.global_forget_accuracy = []
        self.last_accuracy = 0
        
        
    
    def reset(self):
        self.average_accuracy_after_task = []
        self.each_task_accuracy = []
        self.global_forget_accuracy = []
        self.last_accuracy = 0
        
    def average_classification_accuracy(self, model, test_loader):
        
        tasks_accuracies = []
        
        for task_index in range(self.current_task + 1):
            average_accuracy = 0
            
            for data in test_loader[task_index]:
                inputs, labels = data
                inputs = inputs.to(self.args_dict['device'])
                labels = labels.to(self.args_dict['device'])
                
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
                
                metrics = self.calc_metrics(outputs, labels)
                average_accuracy += metrics.item()
            
            tasks_accuracies.append(average_accuracy/len(test_loader[task_index]))
            
        average_accuracy = np.mean(tasks_accuracies)
        
        return average_accuracy
    
    def average_global_forgetting(self, model, test_loader):
        previous_task_accuracy = 0
        
        for data in test_loader[self.current_task -1]:
            inputs, labels = data
            inputs = inputs.to(self.args_dict['device'])
            labels = labels.to(self.args_dict['device'])
            
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
            
            metrics = self.calc_metrics(outputs, labels)
            previous_task_accuracy += metrics.item()
            
        previous_task_accuracy = previous_task_accuracy/len(test_loader[self.current_task - 1])
        
        forget_accuracy = self.each_task_accuracy[self.current_task] - previous_task_accuracy
        return forget_accuracy
    
    

    def calc_metrics(self, pred, target):
        _, pred = torch.max(pred, dim = 1)
        correct = (pred == target).sum()
        
        return correct/target.size(0)
        
    def save_model(self, model_name, dataset_name, epoch, model, task_index):
        folder_name = 'weights-' + str(model_name) + '-' + str(dataset_name) + '-Task-' + str(task_index)
        save_dir = os.path.join(self.args_dict['weights_save_path'], folder_name)
        
        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)
            
        save_name = 'Epoch-' + str(epoch) + '.pth'
        torch.save(model.state_dict(), os.path.join(save_dir, save_name))
    
    def load_model(self, model_name, dataset_name, epoch, model, task_index):
        folder_name = 'weights-' + str(model_name) + '-' + str(dataset_name) + '-Task-' + str(task_index)
        save_dir = os.path.join(self.args_dict['weights_save_path'], folder_name)
        
        load_name = 'Epoch-' + str(epoch) + '.pth'
        model.load_state_dict(torch.load(os.path.join(save_dir, load_name)))
        
        return model

    def calculate_no_of_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    def eval_single(self):
        raise NotImplementedError('Eval single is not implemented')
    
    
    
    def train(self, model, train_loader, optimizer, epoch, replay_loader=None):
    
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
                
        
        ## replay the data
        if replay_loader is not None and epoch % 3 == 0:
            for batch_idx, (data, target) in enumerate(replay_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target) * 0.5
                loss.backward()
                optimizer.step()
              
                
        print(f'Train Epoch: {epoch} Loss: {np.mean(loss_array)} Accuracy: {np.mean(accuracy_array)}')
        with open(file_path, 'a') as f:
            f.write(f'Task Index: {str(self.current_task)}')
            f.write('\n')
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
            
            with open(self.args_dict['record_save_path_cl_tasks'], 'a') as f:
                f.write(f'Model name: ----------- {model_name} --------------')
                f.write('\n')
                f.write(f'Dataset name: ----------- {dataset_name} --------------\n')
                
         
            self.reset()
            
            for task_index in range(self.num_tasks):
                
                
                
                self.current_task = task_index
                
                train_losses = []
                train_accuracies = []
                test_losses = []
                test_accuracies = []
                
                ## load the particular task dataset loader for the train and test data
                train_loader = self.args_dict[('trainloader', dataset_index)][task_index]
                test_loader = self.args_dict[('testloader', dataset_index)][task_index]
                replay_loader = self.args_dict[('replayloader', dataset_index)][task_index] if task_index != 0 else None
                
                print(f'Task Index: {task_index}')
                
                for epoch in range(self.args_dict['epoch_distribution'][task_index]):
                    
                    train_loss, train_accuracy = self.train(model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch, replay_loader=replay_loader)
                    test_loss, test_accuracy = self.evaluate(model=model, test_loader=test_loader, scheduler=scheduler)
                    
                    self.save_model(model_name, dataset_name, epoch, model, task_index)
                    
                    train_accuracies.append(train_accuracy)
                    train_losses.append(train_loss)
                    test_accuracies.append(test_accuracy)
                    test_losses.append(test_loss)
                    
                each_task_accuracy = max(test_accuracies)
                self.each_task_accuracy.append(each_task_accuracy)
                
                incremental_accuracy = self.average_classification_accuracy(model=model, test_loader=self.args_dict[('testloader', dataset_index)])
                self.average_accuracy_after_task.append(incremental_accuracy)
                
                
                if task_index != 0:
                    forget_accuracy = self.average_global_forgetting(model=model, test_loader=self.args_dict[('testloader', dataset_index)])
                    self.global_forget_accuracy.append(forget_accuracy)
               
               
                # # ## change the learning rate 
                # # current_lr = optimizer.param_groups[0]['lr']
                # # new_lr = current_lr / (1*np.exp(-(task_index-2)) )
                
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = new_lr
                
                with open(self.args_dict['record_save_path_cl_tasks'], 'a') as f:
                    f.write("--------------------Task----------------------\n")
                    f.write("Task Index: " + str(task_index))
                    f.write('\n')
                    f.write(f'Incremental Accuracy after task {task_index}: {incremental_accuracy}')
                    f.write('\n')
                    f.write(f'Each Task Accuracy: {each_task_accuracy}')
                    f.write('\n')
                    if task_index != 0:
                        f.write("Forget Accuracy: " + str(forget_accuracy))
                        f.write('\n')
               
   
            
            global_forget_accuracy = np.mean(self.global_forget_accuracy)
            last_accuracy = self.average_accuracy_after_task[-1]
            average_incremental_accuracy = np.mean(self.average_accuracy_after_task)
            
            
            with open(self.args_dict['record_save_path_cl_tasks'], 'a') as f:
                f.write("Avreage Global Forgetting Accuracy: " + str(global_forget_accuracy))
                f.write('\n')
                f.write("Last Task Accuracy: " + str(last_accuracy))
                f.write('\n')
                f.write("Average Incremental Accuracy: " + str(average_incremental_accuracy))
                f.write('\n')
                f.write("-------------------------------------------------\n")
                f.write('\n')
                
        
                
                
            
          
            # self.plot_results(train_losses, train_accuracies, test_losses, test_accuracies, model_name, dataset_name)
                
    
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
        
        
    
    
    def evaluate(self, model, test_loader, scheduler=None):
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
            scheduler.step()
        
        print(f'Validation Loss: {np.mean(loss_array)}')
        print(f'Validation Accuracy: {np.mean(accuracy_array)}')
        print(f'Validation Time: {np.sum(eval_time)}')
            
        with open(self.args_dict['record_save_path'], 'a') as f:
            f.write("Task Index: " + str(self.current_task))
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



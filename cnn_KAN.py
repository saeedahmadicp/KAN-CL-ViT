import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
# from torchsummary import summary
import torch
import numpy as np



from models.resnet import resnet50_v2, ResNet50_Weights

from tqdm import tqdm
from timer import Timer

from efficientkan import KANLinear

# CNN model for CIFAR-10 with KANLinear
class KAN(nn.Module):
    def __init__(self, backbone, num_classes):
        super(KAN, self).__init__()
        self.backbone = backbone
        self.kan1 = KANLinear(2048, 256)  
        #self.kan2 = KANLinear(256, 128)
        self.kan3 = KANLinear(256, num_classes)

    def forward(self, x):
        
        with torch.no_grad():
            x = self.backbone(x)
        
        x = self.kan1(x)
       # x = self.kan2(x)
        x = self.kan3(x)
        return x
    
# CNN model for CIFAR-10 with KANLinear
class MLP(nn.Module):
    def __init__(self, backbone, num_classes):
        super(MLP, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(2048, 256)  
        #self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        
        with torch.no_grad():
            x = self.backbone(x)
        
        x = F.selu(self.fc1(x))
        #x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def print_parameter_details(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()  # Number of elements in the tensor
            total_params += params
            print(f"{name}: {params}")
    print(f"Total trainable parameters: {total_params}") 



# Note the this is just a rough demo for Visualization. Need modifcation. 
def visualize_kan_parameters(kan_layer, layer_name):
    base_weights = kan_layer.base_weight.data.cpu().numpy()
    plt.hist(base_weights.ravel(), bins=50)
    plt.title(f"Distribution of Base Weights - {layer_name}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()
    if hasattr(kan_layer, 'spline_weight'):
        spline_weights = kan_layer.spline_weight.data.cpu().numpy()
        plt.hist(spline_weights.ravel(), bins=50)
        plt.title(f"Distribution of Spline Weights - {layer_name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.show()



def train(model, device, train_loader, optimizer, epoch, file_path):
    
    loss_array = []
    accuracy_array = []
    epoch_time = []
    timer = Timer()
    
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
        accuracy = output.argmax(dim=1).eq(target).float().mean().item()
        accuracy_array.append(accuracy)
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
        
        
        

def evaluate(model, device, test_loader, file_path):
    model.eval()
    test_loss = 0
    correct = 0
    epoch_time = []
    loss_array = []
    timer = Timer()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            timer.start()
            output = model(data)
            timer.stop()
            epoch_time.append(timer.get_elapsed_time())
            test_loss = nn.CrossEntropyLoss()(output, target).item()
            loss_array.append(test_loss)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
  
    
    print(f'\nTest set: Average loss: {np.mean(loss_array):.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    
    with open(file_path, 'a') as f:
        f.write("Test set results:\n")
        f.write(("Average test loss: "+ str(np.mean(loss_array))))
        f.write('\n')
        f.write(("Accuracy: " + str(correct) + "/" + str(len(test_loader.dataset)) + " (" + str(100. * correct / len(test_loader.dataset)) + "%)"))
        f.write('\n')
        f.write(f'Epoch Time: {np.sum(epoch_time)}')
        f.write('\n')
        f.write("Average time per iteration: " + str(np.mean(epoch_time)))
        f.write('\n')
        f.write("-------------------------------------------------\n")
        
        

def calculate_no_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = CNN().to(device) 

    # Uncommnet this line for CNN KAN. 
    # model = CNNKAN().to(device) 
    # print(model) 
    # print_parameter_details(model)
    # summary(model,  input_size=(3, 32, 32))
    
    ## ResNet50 pretrained model backbone
    resnet_50 = resnet50_v2(weights= ResNet50_Weights.IMAGENET1K_V2)#"IMAGENET1K_V2")
    resnet_50.fc = nn.Identity()

    # Freezing the layers of the resnet model
    for param in resnet_50.parameters():
        param.requires_grad = False
        
    
    model_MLP = MLP(backbone=resnet_50, num_classes=10).to(device=device) #CNN().to(device)
    model_KAN = KAN(backbone=resnet_50, num_classes=10).to(device=device)   #CNNKAN().to(device)
   # model_KAN_original = CNNKAN_original(device=device).to(device)
    
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  
    ])
    
    ## batch size
    batch_size = 512 #20 #500

    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    file_path = 'saved_models\\KAN_vs_MLP.txt'
    
    models = [model_MLP, model_KAN]
    model_names = ['MLP_with_resnet50_backbone', 'KAN_with_resnet50_backbone']
    dataset_name = 'CIFAR10'
    
    
    for model, model_name in zip(models, model_names):
        for name, param in model.named_parameters():
            print(f"{name}: {param.size()} {'requires_grad' if param.requires_grad else 'frozen'}")

        # TODO: Need to explore various Optimizer and optimize the Learning Rate.
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        
        total_params = calculate_no_of_parameters(model)
    
        with open(file_path, 'a') as f:
                f.write(f'Model name: ----------- {model_name} --------------')
                f.write('\n')
                f.write(f'Dataset name: ----------- {dataset_name} --------------\n')
                f.write(f'Total trainable parameters: {total_params}' + '\n')

        for epoch in range(15):
            train(model, device, train_loader, optimizer, epoch, file_path)
            evaluate(model, device, test_loader, file_path)
            
        
        with open(file_path, 'a') as f:
            f.write('\n\n\n')
            f.write('----------------------------------------------------')
            f.write('\n\n\n')
        
    # torch.save(model.state_dict(), 'model_weights_KAN.pth')
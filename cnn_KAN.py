# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# # from torchsummary import summary
# import torch



# from models.resnet import resnet50_v2, ResNet50_Weights



# from efficientkan import  KAN as efficientKAN
# from fastkan import FastKAN as fastKAN
# # CNN model for CIFAR-10 with KANLinear
# class EfficientKAN(nn.Module):
#     def __init__(self, backbone, num_classes):
#         super(EfficientKAN, self).__init__()
#         self.backbone = backbone
#         self.efficientKAN = efficientKAN([2048, 256, num_classes])

#     def forward(self, x):
        
#         with torch.no_grad():
#             x = self.backbone(x)
        
#         x = self.efficientKAN(x)
#         return x
    
# # CNN model for CIFAR-10 with KANLinear
# class MLP(nn.Module):
#     def __init__(self, backbone, num_classes):
#         super(MLP, self).__init__()
#         self.backbone = backbone
        
#         self.mlp = nn.Sequential(
#             nn.Linear(2048, 256),
#             nn.SELU(),
#             nn.Linear(256, num_classes)
#         )
        
#     def forward(self, x):
#         with torch.no_grad():
#             x = self.backbone(x)
#         x = self.mlp(x)
#         return x
    
    
# # CNN model for CIFAR-10 with fastKAN
# class FastKAN(nn.Module):
#     def __init__(self, backbone, num_classes):
#         super(FastKAN, self).__init__()
#         self.backbone = backbone
#         self.fastKAN = fastKAN([2048, 256, num_classes])
        
#     def forward(self, x):
#         with torch.no_grad():
#             x = self.backbone(x)
#         x = self.fastKAN(x)
#         return x


# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     #model = CNN().to(device) 

#     # Uncommnet this line for CNN KAN. 
#     # model = CNNKAN().to(device) 
#     # print(model) 
#     # print_parameter_details(model)
#     # summary(model,  input_size=(3, 32, 32))
    
#     ## ResNet50 pretrained model backbone
#     resnet_50 = resnet50_v2(weights= ResNet50_Weights.IMAGENET1K_V2)#"IMAGENET1K_V2")
#     resnet_50.fc = nn.Identity()

#     # Freezing the layers of the resnet model
#     for param in resnet_50.parameters():
#         param.requires_grad = False
        
    
#     model_MLP = MLP(backbone=resnet_50, num_classes=10).to(device=device) #CNN().to(device)
#     model_KAN = KAN(backbone=resnet_50, num_classes=10).to(device=device)   #CNNKAN().to(device)
#    # model_KAN_original = CNNKAN_original(device=device).to(device)
    
    
    
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  
#     ])
    
#     ## batch size
#     batch_size = 512 #20 #500

    
#     train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
#     file_path = 'saved_models\\KAN_vs_MLP.txt'
    
#     models = [model_MLP, model_KAN]
#     model_names = ['MLP_with_resnet50_backbone', 'KAN_with_resnet50_backbone']
#     dataset_name = 'CIFAR10'
    
    
  
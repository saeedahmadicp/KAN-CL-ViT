import matplotlib.pyplot as plt
import os



AIA_MLP_ViT = [0.489, 0.218, 0.149, 0.119, 0.093, 0.087, 0.074, 0.044, 0.042, 0.048]

AIA_KAN_ViT = [0.550, 0.271, 0.168, 0.135, 0.108, 0.096, 0.083, 0.049, 0.042, 0.048]

AIA_MLP = [0.9931640625, 0.4997702205882353, 0.326763117313385, 0.24805501302083333, 0.20924479166666665]
AIA_KAN = [0.9951171875, 0.5019818474264706, 0.33346693165162034, 0.26481911256982416, 0.4650079884285144]


plt.plot(AIA_KAN_ViT, label='AIA_KAN_Vit')
plt.plot(AIA_MLP_ViT, label='AIA_MLP')
plt.ylabel('Average Incremental Accuracy (AIA)')
plt.xlabel('No of Tasks')
plt.legend()
file_name = 'AIA_KAN_vs_MLP.png'
plt.savefig(file_name)
plt.show()



plt.plot(AIA_KAN, label='AIA_KAN')
plt.plot(AIA_MLP, label='AIA_MLP')
plt.ylabel('Average Incremental Accuracy (AIA)')
plt.xlabel('No of Tasks')
plt.legend()
file_name = 'AIA_KAN_vs_MLP.png'
plt.savefig(file_name)
plt.show()



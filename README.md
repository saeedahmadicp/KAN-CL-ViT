## Investigating the Strengths and Limitations of KAN in Continual Learning

### Description
---
<p align="justify">
This project investigates the performance of different variants of Kolmogorov-Arnold Networks (KAN) for image classification tasks in a continual learning setting. We conducted two sets of experiments: one using standalone Multi-Layer Perceptrons (MLPs) and KAN variants, and another integrating KAN into the Vision Transformer (ViT) architecture.
</p>

<p align="justify">
The experiments were carried out on the MNIST and CIFAR100 datasets, which were divided into multiple tasks to simulate a continual learning scenario. The datasets were split as follows:
</p>

#### MNIST Experiments (MLP and KAN)

- Total Classes: 10
- Number of Tasks: 5
- Classes per Task: 2
- Epochs for Task 1: 7
- Epochs for Remaining Tasks: 5

#### CIFAR100 Experiments (ViT with MLP and KAN)

- Total Classes: 100
- Number of Tasks: 10
- Classes per Task: 10
- Epochs for Task 1: 25
- Epochs for Remaining Tasks: 10
<br>

<p align="justify">
The primary objective was to investigate the strengths and limitations of KAN in a continual learning setting, where the model must learn new tasks while retaining knowledge from previously learned tasks. By comparing the performance of KAN variants with traditional MLPs and integrating KAN into the ViT architecture, we aimed to gain insights into the potential advantages and drawbacks of using KAN for continual learning tasks.
</p>


### Key Findings
---
Here are the findings of our project:
- In the case of standalone MLP and KAN experiments, the KAN model demonstrates superior performance on the CL task and shows better resistance to forgetting the previous knowledge while learning the new task.
- While for the KAN-ViT, there was a slight improvement in the overall average incremental accuracy, especially in the early incremental stages, however, in later stages, the performance of both the MLP and KAN-ViT remains the same, as demonstrated by the below graph

<p align="center"> <img align="center" src="https://github.com/saeedahmadicp/KAN-CL-ViT/blob/main/results/ViT.png" alt="Performance Graph"> </p>

### Future Work
---
Based on the findings and observations from this project, several potential future directions and improvements can be explored:
- Running the experiments across more complex datasets to further evaluate the performance and scalability of KAN in continual learning scenarios.
- Building on top of the base implementation of KAN and introducing replay mechanisms to further minimize the impacts of forgetting and improve incremental learning capabilities.
- Experimenting with various parameter regularization techniques to build on the promise of KANs to mitigate catastrophic forgetting.
- Using proper scheduling strategies to ensure consistent learning across all incremental stages.
- CL and incremental learning can be further improved by selectively updating and masking the activations and neurons to mitigate catastrophic forgetting.

By exploring these avenues, researchers can potentially unlock the full potential of KAN for continual learning tasks and contribute to the advancement of this field.

---
References
---
<p align="justify">
[1] Liu, Ziming, et al. "Kan: Kolmogorov-arnold networks." arXiv preprint arXiv:2404.19756 (2024). <br>
[2] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).


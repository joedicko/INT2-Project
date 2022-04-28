import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

training_data = datasets.CIFAR10(
    root = "data",
    train = True,
    download = True,
    # transform = ToTensor()
)

labels_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

for i in range(16):
    random_index = np.random.randint(len(training_data))
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_data[random_index][0], cmap=plt.cm.binary)
    plt.xlabel(labels_map[training_data[random_index][1]])
plt.show()
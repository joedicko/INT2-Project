import torch
from torchvision import datasets
from torchvision.transforms import Lambda
import torchvision.transforms as tt
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.ToTensor(),
                         tt.Normalize(*stats, inplace=True)])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

# Imports the training dataset
training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tfms, target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

# Imports the testing dataset
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=valid_tfms)

"""Labels Map"""

# Creates a dictionary that allows for key mapping to strings
labels_map = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""Iterating and Visualising the Dataset

Preparing DataLoaders
"""

# Dataloaders present the data in batches to be trained on, rather than the
#   entire dataset.
# This is beneficial as the back propogation can occur more regularly, and
#   it also prevents overfitting
# Dataloaders are also useful as the libraries we are using work better on
#   batches of data rather than singles or entire datasets
train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

"""Iterating Through DataLoader

Device for Training
"""

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

"""# Defining Neural Network Class"""


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvStack = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.Flatten = nn.Flatten()
        self.LinearStack = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.ConvStack(x)
        x = self.Flatten(x)
        # x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        # print(x)
        x = self.LinearStack(x)
        return x


model = NeuralNetwork().to(device)
print(model)

"""Define Loss Function and Optimizer"""

criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

"""Training the Network"""

# number of epochs to train the model
n_epochs = 50
# List to store loss to visualize
train_losslist = []
test_losslist = []
test_loss_min = 1000000  # track change in validation loss

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    test_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimiser.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimiser.step()
        # update training loss
        train_loss += loss.item() * inputs.size(0)
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'Training - [{epoch + 1}, {i + 1:5d}] loss: {train_loss / 2000:.3f}')

    ######################
    # validate the model #
    ######################
    model.eval()
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        output = model(inputs)
        loss = criterion(output, labels)
        # update average test loss
        test_loss += loss.item() * inputs.size(0)
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #    print(f'Testing - [{epoch + 1}, {i + 1:5d}] loss: {train_loss / 2000:.3f}')

    # calculate average loss
    train_loss = train_loss / len(train_dataloader.dataset)
    test_loss = test_loss / len(test_dataloader.dataset)
    # put loss in list, to be shown in graph
    train_losslist.append(train_loss)
    test_losslist.append(test_loss)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
        epoch, train_loss, test_loss))

    # save model if validation loss has decreased
    if test_loss <= test_loss_min:
        print('New best model found ({:.6f} --> {:.6f}), saving'.format(
            test_loss_min,
            test_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        test_loss_min = test_loss

"""Saving the model"""

PATH = './model_cifar.pth'
torch.save(model.state_dict(), PATH)

model = NeuralNetwork()
model.load_state_dict(torch.load('model_cifar.pt', map_location=torch.device('cpu')))
model.eval()

"""Testing on Test Data"""

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    model = model.cuda()
    for data in test_dataloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # debug ### print(outputs.data)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # debug ### print(predicted.shape, labels.shape)
        # debug ### print(predicted)
        # debug ### print(labels)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

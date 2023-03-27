import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.core.dataset import Dataset
from torch.utils.data import DataLoader

# get command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')
parser.add_argument("--num_epochs", type=int, default=25, help="number of epochs to train")
parser.add_argument("--output_dir", type=str, help="output directory")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
args = parser.parse_args()

# get the Azure ML run object
run = Run.get_context()

# get the workspace
ws = run.experiment.workspace

# get the datastore to download data
datastore = Datastore.get(ws, datastore_name='Snapshot_Serengeti')


# dataset get by name
dataset = Dataset.get_by_name(ws, name='Snapss')
dataset =dataset.download(target_path='.', overwrite=False)

# get the training and testing dataset
train_set = torchvision.datasets.ImageFolder(root="./train", transform=transforms.ToTensor())
test_set = torchvision.datasets.ImageFolder(root="./val", transform=transforms.ToTensor())


# Load the training and test data
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

# load the pre-trained DenseNet 201 model
model = torchvision.models.densenet201(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 50)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# define the loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# train the model
for epoch in range(args.num_epochs):
    print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                run.log('training loss', running_loss / 100)
                running_loss = 0.0
        
    print('Finished Training')
    
# save the model
os.makedirs(args.output_folder, exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.output_folder, 'model.pth'))

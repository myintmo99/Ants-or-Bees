import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

mean = np.array([0.5,0.5,0.5])
std = np.array([0.25,0.25,0.25])

# Define Data Transformation
data_transform = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
}

data_dir = 'D:/Data D/Online Learning/Pytorch/hymenoptera_data'
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x),data_transform[x]) 
                  for x in ['train','val']}
dataloader = {x : torch.utils.data.DataLoader(dataset=image_datasets[x] , batch_size = 4, shuffle=True, num_workers=0) 
              for x in ['train','val']}

datasets_size = {x : len(image_datasets[x]) for x in ['train','val']}
class_names = image_datasets['train'].classes
print(class_names)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image, label = next(iter(dataloader['train']))

# image = torchvision.utils.make_grid(image)
# image = image.numpy().transpose((1, 2, 0)) # change shape from (3,228,906) to (228,906,3)
# image = std*image+mean
# image = np.clip(image,0,1)
# print(image.shape)
# plt.imshow(image)
# plt.show()

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('_'*10)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss/datasets_size[phase]
            epoch_acc = running_corrects.double()/datasets_size[phase]

            print('{} Loss: {:.4f}, Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc>best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time()-start # Time Taken for Training
    print('Training complete in {:.0f}m and {:.0f}s'.format(time_elapsed//60 ,time_elapsed%60))
    print('Best Val Acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

model = resnet18(weights=ResNet18_Weights.DEFAULT)
# Freezing all layers and training just the final layer
# for param in model.parameters():
#     param.requires_grad=False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) # Redefine the final layer to suit your model (will also change the requires_grad to True as default)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# adjusting the learning rate during training
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=2)
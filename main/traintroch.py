import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import cv2
import torchvision

import torchvision.models as models
model = models.resnet18(pretrained=True)

test_dir = '../images/testing'
train_dir = '../images/training'

train_files = []

for subfolder in os.listdir(train_dir):
    print(subfolder)
    if subfolder == '.DS_Store':
        continue
    for files in os.listdir(os.path.join(train_dir, subfolder)):
        if files == '.DS_Store':
            continue
        train_files.append(os.path.join(subfolder, files))

test_files = []
for subfolder in os.listdir(test_dir):
    #print(subfolder)
    if subfolder == '.DS_Store':
        continue
    for files in os.listdir(os.path.join(test_dir, subfolder)):
        if files == '.DS_Store':
            continue
        test_files.append(os.path.join(subfolder, files))

print(len(train_files))
print(len(test_files))

labels = ["glass", "paper", "metal", "plastic"]

# inheretence from the pytorch dataset 
class WasteDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform = None):
        self.file_list = file_list
        self.dir = dir
        self.mode= mode
        self.transform = transform
        if self.mode == 'train':
            for i, label in enumerate(labels):
                if label in self.file_list[0]:
                    # label is in number rather than an array
                    self.label = i
            
    def __len__(self):
        return len(self.file_list)
    
    # override get item method, get the file input and the label of each item
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]
        
# define data augmentation
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])

#Transform the test dataset
test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# generate dataset
class_files = [0] * len(labels)
for i, label in enumerate(labels):
    class_files[i] = [fileName for fileName in train_files if label in fileName]

train_datasets = [0] * len(labels)
for i in range(len(labels)):
    train_datasets[i] = WasteDataset(class_files[i], train_dir,  transform=data_transform)

class_files = [0] * len(labels)
for i, label in enumerate(labels):
    class_files[i] = [fileName for fileName in test_files if label in fileName]

test_datasets = [0] * len(labels)
for i in range(len(labels)):
    test_datasets[i] = WasteDataset(class_files[i], test_dir, mode='test', transform=test_transform)

trainset = ConcatDataset(train_datasets)
train_loader = DataLoader(trainset, batch_size = 64, shuffle=True, num_workers=4)

testset = ConcatDataset(test_datasets)
test_loader = DataLoader(testset, batch_size = 64, shuffle=False, num_workers=4)


# show some pictures from the dataset
samples, labellist = iter(train_loader).next()
#plt.figure(figsize=(16,24))
grid_imgs = torchvision.utils.make_grid(samples[:24])

print(labellist[:24])

np_grid_imgs = grid_imgs.numpy()
print(np_grid_imgs.shape)
# in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
img = np.transpose(np_grid_imgs, (1, 2, 0))
img = cv2.resize(img, (224 * 8, 224 * 3), interpolation=cv2.INTER_CUBIC)
cv2.imshow("img", img)
cv2.waitKey()
#plt.imshow()
#plt.waitforbuttonpress()

# Freeze the parameters, so they won't get update
for param in model.parameters():
    param.require_grad = False

# Override the fc layers from the original dataset
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(512, 100)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(100, 2)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.fc = fc

# train the model 
def train(model, trainloader, testloader, criterion, optimizer, epochs = 5):
    train_loss =[]
    train_len = len(trainloader)
    val_len = len(testloader)

    for e in range(epochs):
        running_loss = 0
        correct_items_train = 0
        correct_items_val = 0

        for images, labellist in trainloader:
            #inputs, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            img = model(images)

            #train accuracy
            prediction = torch.argmax(img, dim=1)
            label = torch.argmax(labellist, dim=1)
            if prediction == label:
                correct_items_train += 1

            loss = criterion(img, labellist)
            running_loss+=loss
            loss.backward()
            optimizer.step()

        print("Epoch : {}/{} ".format(e+1,epochs), 
        "Training Loss: {:.6f} ".format(running_loss/train_len),
        "Training Accuracy: {:.6f} \n".format(correct_items_train/train_len))

        # validation accuracy
        for images, labes in testloader: 
            output = model(x)
            prediction = torch.argmax(img, dim=1)
            label = torch.argmax(labellist, dim=1)
            if prediction == label:
                correct_items_val += 1

        print("Validation Accuracy: {:6f} \n".format(correct_items_val/val_len))

        train_loss.append(running_loss)
    
    plt.plot(train_loss,label="Training Loss")
    plt.show() 


epochs = 3
model.train()
# optimizer input -> parameters to optimize and learning rate
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.NLLLoss()   # cross-entropy loss
train(model,train_loader,test_loader,criterion, optimizer, epochs)

#Save the model
filename_pth = 'ckpt_resnet18_catdog.pth'
torch.save(model.state_dict(), filename_pth)


model.eval()
#fn_list = []
#pred_list = []
correct_items_test = 0
for img, labellist in test_loader:
    with torch.no_grad():
        #x = x.to(device)
        output = model(img)
        prediction = torch.argmax(output, dim=1)
        label = torch.argmax(labellist, dim=1)
        if prediction == label:
            correct_items_test += 1
        #fn_list += [n[:-4] for n in fn]
        #pred_list += [p.item() for p in pred]

print("Testing Accuracy: {:6f}".format(correct_items_test/len(test_loader)))

#submission = pd.DataFrame({"id":fn_list, "label":pred_list})
#submission.to_csv('preds_resnet18.csv', index=False)


samples, label = iter(test_loader).next()

#samples = samples.to(device)
fig = plt.figure(figsize=(24, 16))
fig.tight_layout()

output = model(samples[:24])
prediction = torch.argmax(output, dim=1)
prediction = [p.item() for p in prediction]

for i, sample in enumerate(samples[:24]):
    #plt.subplot(4,6,num+1)
    plt.title(labels[prediction[i]])
    plt.axis('off')
    sample = sample.numpy()
    plt.imshow(np.transpose(sample, (1,2,0)))



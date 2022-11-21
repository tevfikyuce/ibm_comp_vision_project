import os
import torch
from torch.utils.data import Dataset
import torch.utils.data
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

#Creating Custom DataSet Class
class stopNotStopData(Dataset):
    def __init__(self, stop_dir, not_stop_dir, transform=None):
        self.stop_dir = stop_dir
        self.not_stop_dir = not_stop_dir

        #Finding Number of Stop and Non-Stop Images
        self.stop_image_list =  os.listdir(self.stop_dir)
        self.not_stop_image_list =  os.listdir(self.not_stop_dir)
        self.n_stop_images = len(self.stop_image_list)
        self.n_not_stop_images = len(self.not_stop_image_list)
        self.n_images = self.n_not_stop_images + self.n_stop_images

        #Determining the reshaping values of each image
        self.width_distr = np.zeros(self.n_images)
        self.heigth_distr = np.zeros(self.n_images)
        for i in range(len(self.stop_image_list)):
            stop_image_name = self.stop_image_list[i]
            temp_image = Image.open(self.stop_dir + stop_image_name)
            self.width_distr[i] = temp_image.width
            self.heigth_distr[i] = temp_image.height

        for i in range(len(self.not_stop_image_list)):
            not_stop_image_name = self.not_stop_image_list[i]
            temp_image = Image.open(self.not_stop_dir + not_stop_image_name)
            self.width_distr[i+len(self.stop_image_list)] = temp_image.width
            self.heigth_distr[i+len(self.stop_image_list)] = temp_image.height

        #Calculating means as target widths and heights
        self.reshape_width = np.round_(np.mean(self.width_distr)).astype(int)
        self.reshape_heigth = np.round_(np.mean(self.heigth_distr)).astype(int)

        #Creating initial labels for images
        self.labels = np.zeros(self.n_images)

        self.transform = transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        #First find the corresponding image
        if idx+1<=self.n_stop_images:
            image_name = self.stop_image_list[idx]
            image_path = os.path.join(self.stop_dir, image_name)
            #Image is a stop image first load it
            #image = Image.open(self.stop_dir+str(int(idx+1))+".JPG")
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.reshape_width, self.reshape_heigth), resample=Image.BICUBIC)
            #image = np.asarray(image).astype(np.uint8)
            label = 0 # 0 is for stop image and 1 is for non-stop image
            self.labels[idx] = 0 #Set it also in labels array
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(label)
        else:
            idx = idx - self.n_stop_images
            image_name = self.not_stop_image_list[idx]
            image_path = os.path.join(self.not_stop_dir, image_name)
            #image = read_image(self.not_stop_dir+str(int(idx+1))+".JPG")
            #image = Image.open(self.not_stop_dir+str(int(idx+1))+".JPG")
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.reshape_width, self.reshape_heigth), resample=Image.BICUBIC)
            #image = np.asarray(image).astype(np.uint8)
            label = 1 # 0 is for stop image and 1 is for non-stop image
            self.labels[idx] = 1 #Set it also in labels array
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(label)

        return image, label

#Creating CNN class
#Test of the training part for basic structure
class stopNotStop_cnn(nn.Module):
    #Constructor
    def __init__(self, out_channel_1 = 16, out_channel_2 = 32):
        super(stopNotStop_cnn, self).__init__()
        self.out_channel_1 = out_channel_1
        self.out_channel_2 = out_channel_2
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.out_channel_1, kernel_size=5, padding=3, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=self.out_channel_1, out_channels=self.out_channel_2, kernel_size=3, padding=1, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)

        self.linear1 = nn.Linear(out_channel_2*43*31, 1024)
        self.linear2 = nn.Linear(1024, 1)

        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x

#Definition of dataset splitting function
def split_dataset(dataset, fraction_arr, random_seed=75):
    #First Find Number of Elements of each dataset
    num_elements = np.round_([len(dataset)*fraction_arr[0], len(dataset)*fraction_arr[1], len(dataset)*fraction_arr[2]]).astype(int)
    num_elements[0] = num_elements[0] - (np.sum(num_elements)-len(dataset))
    #Split dataset into three parts
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, num_elements, generator=torch.Generator().manual_seed(random_seed))
    return train_dataset, valid_dataset, test_dataset

def train_model(model, device, n_epochs, train_loader, validation_loader, optimizer, criterion, N_test, decision_threshold=0.5):
    cost_list = []
    accuracy_list = []
    # Loops for each epoch
    for epoch in range(n_epochs):
        # Keeps track of cost for each epoch
        COST=0
        # For each batch in train loader
        for x, y in train_loader:
            x = x.to(device)
            y = y.unsqueeze(1)
            y = y.to(device)
            # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            optimizer.zero_grad()
            # Makes a prediction based on X value
            z = model(x)
            # Measures the loss between prediction and acutal Y value
            z = z.to(torch.float32)
            y = y.to(torch.float32)
            loss = criterion(z, y)
            # Calculates the gradient value with respect to each weight and bias
            loss.backward()
            # Updates the weight and bias according to calculated gradient value
            optimizer.step()
            # Cumulates loss 
            COST+=loss.data
        
        # Saves cost of training data of epoch
        cost_list.append(COST)
        # Keeps track of correct predictions
        correct=0
        # Perform a prediction on the validation  data  
        for x_test, y_test in validation_loader:
            y_test = y_test.to(device)
            x_test = x_test.to(device)

            # Makes a prediction
            z = model(x_test)
            
            #Make prediction and compare with actual result
            predict = (z > decision_threshold)
            predict = torch.squeeze(predict.long())
            # Checks if the prediction matches the actual value
            correct += (predict == y_test).sum().item()
        
        # Calcualtes accuracy and saves it
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
        print('Epoch ' + str(epoch+1) + 'Finished!')
    return model, cost_list, accuracy_list

def plot_prediction(model, x_test):
    x_test = torch.unsqueeze(x_test, dim=0)

    #Making prediction
    z = model(x_test)

    x_test = torch.squeeze(x_test, dim=0)
    x_test = x_test.moveaxis(0,-1)

    plt.imshow(x_test)
    if z < 0.5:
        plt.title('Prediction: STOP')
    else:
        plt.title('Prediction: NOT-STOP')
    plt.show()


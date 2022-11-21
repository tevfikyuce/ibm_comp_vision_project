import class_and_functions
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np

#Dataset Parameters
image_transform = transforms.ToTensor()
stop_dir = './Data/stop/'
not_stop_dir = './Data/not_stop/'
dataset = class_and_functions.stopNotStopData(stop_dir=stop_dir, not_stop_dir=not_stop_dir, transform=image_transform)

#Splitting Dataset into three datasets
fraction_arr = [0.8, 0.1, 0.1] #Fraction of training, validation and test respectively
train_set, valid_set, test_set = class_and_functions.split_dataset(dataset=dataset, fraction_arr=fraction_arr, random_seed=25)

#Select the device
device = "cpu"

#Defining hyperparameter lists to test
lr_list = np.asarray([0.1, 0.01])
n_epochs_list = np.asarray([5, 10]).astype(int)
out_channel_1_list = np.asarray([2, 4]).astype(int)
out_channel_2_list = np.asarray([4, 8]).astype(int)
train_batch_size_list = [16, 32]

best_acc = 0 #Best Accuracy is initially zero

for lr in lr_list:
    for n_epochs in n_epochs_list:
        for out_channel_1 in out_channel_1_list:
            for out_channel_2 in out_channel_2_list:
                for train_batch_size in train_batch_size_list:
                    model = class_and_functions.stopNotStop_cnn(out_channel_1=out_channel_1, out_channel_2=out_channel_2).to(device)

                    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                    criterion = nn.BCELoss()

                    train_dataloader = DataLoader(dataset=train_set, batch_size=train_batch_size ,shuffle=True)
                    valid_dataloader = DataLoader(dataset=valid_set, batch_size=len(valid_set), shuffle=True)
                    N_test = len(valid_set)
                    model, cost_arr, acc_arr = class_and_functions.train_model(model=model, n_epochs=n_epochs, train_loader=train_dataloader, validation_loader=valid_dataloader, optimizer=optimizer, criterion=criterion, N_test=N_test, device=device)
                    current_acc = acc_arr[-1] #Take final accuracy 
                    print('lr='+str(lr) + ' n_epochs=' + str(n_epochs) + ' out_ch_1=' + str(out_channel_1) + ' out_ch2=' + str(out_channel_2) + ' batch='+str(train_batch_size))
                    print('Accuracy = ' + str(current_acc))

                    if current_acc > best_acc:
                        best_model = model
                        best_acc = current_acc

#Print and save the final model
print('Best Model Accuracy: ' + best_acc)
torch.save(best_model, './Model/best_model_acc'+str(best_acc))
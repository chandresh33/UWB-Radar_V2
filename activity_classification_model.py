import os
from scipy.io import wavfile
import numpy as np
import csv
import pandas as pd
import tqdm.notebook as tqdm
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.transforms import transforms


class Activ_CNN(nn.Module):
    def __init__(self):
        super(Activ_CNN, self).__init__()

        # 59049 x 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 19683 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 6561 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 2187 x 128
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 729 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 243 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(0.5))
        # 81 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 27 x 256
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 9 x 256
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 3 x 256
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 1 x 512
        self.conv11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5))
        # 1 x 512
        self.fc = nn.Linear(512, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width

        x = x.view(x.shape[0], 1, -1)
        # x : 23 x 1 x 59049

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)

        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)
        # logit = self.activation(logit)

        return logit


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, param):
        out = self.layer1(param)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class ActivityDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        sig_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        fs, sig_og = open_wav(sig_path)
        sig = copy.deepcopy(sig_og)
        sig = torch.from_numpy(sig)

        y_label_act = torch.tensor(int(self.annotations.iloc[index, 1]))
        y_label_mot = torch.tensor(int(self.annotations.iloc[index, 2]))

        # if self.transform:
        #     sig = self.transform(sig)

        return [sig, y_label_act, y_label_mot]

def open_wav(full_path):
    return wavfile.read(full_path)


def save_dict2csv(data_dir, f_name):
    full_data = {}
    motion_keys = {"sl": 1, "sm": 2, "ml": 3}
    activity_keys = {"squ": 1, "sta": 2}
    file_list, activity_list, motion_list = [], [], []

    for file in os.listdir(data_dir):
        activity_target = file[0:3]
        motion_target = file[-6:-4]

        if motion_target in motion_keys.keys():
            motion_target = motion_keys[motion_target]

        if activity_target in activity_keys.keys():
            activity_target = activity_keys[activity_target]

        file_list.append(file)
        activity_list.append(activity_target)
        motion_list.append((activity_target*10)+motion_target)

    full_data["file"] = file_list
    full_data["Activity"] = activity_list
    full_data["Motion"] = motion_list

    full_data = pd.DataFrame(data=full_data)
    full_data.to_csv(f_name, index=False, header=True)

    return full_data


def open_csv2dict(file_path):
    with open(file_path, 'wb') as f:
        w = csv.writer(f)
        w.writerow(somedict.keys())
        w.writerow(somedict.values())

    return w


def train_model(model_, train_loader_, epochs_):
    losses = []
    training_acc = []
    running_loss = 0
    correct = 0
    total = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for e in tqdm.tqdm((range(epochs_))):
        print("Training for ", e, "epochs.")
        for sigs, labels, we in train_loader_:
            labels -= 1
            sigs, labels = sigs.float().to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model_(sigs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sigs[0].size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        losses.append(running_loss / len(train_loader_))
        training_acc.append(correct/total)

    return model_


def model_tester(model_, test_set_):
    model_.eval()
    # model_.to(device)
    criterion = nn.CrossEntropyLoss()

    correct, total, running_loss = 0, 0, 0
    with torch.no_grad():
        for sigs, labels, we in test_set_:
            labels -= 1
            sigs, labels = sigs.float().to(device), labels.to(device)
            sigs = torch.transpose(torch.unsqueeze(sigs, 0), 0, 1)
            outputs = model_(sigs)

            test_loss = criterion(outputs, labels).to(device)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += test_loss.item() * sigs.size(0)

    accuracy = correct / (total+1e-09)
    epoch_loss = running_loss / (len(test_loader)+1e-09)

    return accuracy, epoch_loss

## dataset paths
path = "Processed_data\\"
train_path, test_path = path+"Train_set\\", path+"Test_set\\"
csv_name = "dataframe.csv"

device = "cpu"

# Initialise the network
model = Activ_CNN()
# model = model.to(device)

# These method are for saving the data labels as csv files
# save_dict2csv(train_path, path+"Train_"+csv_name)
# save_dict2csv(test_path, path+"Test_"+csv_name)

# testing_train = pd.read_csv(path+"Train_dataframe.csv")
# testing_train = pd.read_csv(path+"Test_dataframe.csv")
# print(testing_train.iloc[0, 0], testing_train.iloc[0, 1], testing_train.iloc[0, 2])

train_set = ActivityDataset(csv_file=path+"Train_dataframe.csv", root_dir=train_path, transform=transforms.ToTensor())
test_set = ActivityDataset(csv_file=path+"Test_dataframe.csv", root_dir=test_path, transform=transforms.ToTensor())

batch_size = 10
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()

# Test the initial network configurations (expect low accuracy as it has not been optimised)
# t_vals = model_tester(model, test_loader)
# print("The initialised model: accuracy = {}, losss =  {}".format(t_vals[0], t_vals[1]))
# print("\n")

# Training the network (start with 30 epochs)
training_epochs = 10
# print(model.conv1[0].weight[0][0])
trained_model = train_model(model, train_loader, training_epochs)
# print(model.conv1[0].weight[0][0])
# print("\n")

# # Testing the trained network on the test set
t_vals = model_tester(trained_model, test_loader)
print("The {} epoch trained model: accuracy = {}, losss =  {}".format(training_epochs, t_vals[0], t_vals[1]))
print("\n")

# model.eval()
# print(model(sig))

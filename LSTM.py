import os
z = os.path.abspath("")
y = os.getcwd()
print(y)
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('test_data_copy_of_eeg_file.csv')

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length), :-1]  # All columns except the last one are features
        y = data[i+seq_length-1, -1]  # The last column is the target
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def replace_value(data):
    for i in range(1, len(data)):
        print(data)
        print(typeof(data))
        a = data[i]
        print(a)
        print(typeof(a))
        if a is not None:
            data[i] = 1
        else:
            data[i] = 0
    return data
        
eeg_data = df.iloc[:, 1:-1].values
keypress_data = df.iloc[:, -1].values

scaler = StandardScaler()

eeg_normalized = scaler.fit_transform(eeg_data)

seq_length = 12
eeg_seq, keystroke_seq = create_sequences(np.hstack((eeg_normalized, keypress_data.reshape(-1,1))), seq_length)

eeg_tensor = torch.tensor(eeg_seq, dtype=torch.float32)
keystroke_tensor = torch.tensor(keystroke_seq, dtype=torch.float32)

eeg_train, eeg_test, keystroke_train, keystroke_test = train_test_split(eeg_tensor, keystroke_tensor, test_size=0.2, random_state=47)

batch_size = 50
train_data = TensorDataset(eeg_train, keystroke_train)
test_data = TensorDataset(eeg_test, keystroke_test)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fully_connected = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        initial = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_init = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (initial, c_init))
        out = self.fully_connected(out[:, -1, :])
        return out

input_size = 8 #eeg_train.shape[1]
hidden_size = 128
num_layers = 2
num_classes = 1

model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

crit = nn.BCEWithLogitsLoss() #for binary classification
optim = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000

for epoch in range(num_epochs):
    for i, (features,labels) in enumerate(train_loader):
        outputs = model(features)
        loss = crit(outputs.squeeze(), labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

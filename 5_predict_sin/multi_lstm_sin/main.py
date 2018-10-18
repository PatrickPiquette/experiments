import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from simplesin import SimpleSin
from pplstm import PPLSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

n_epochs = 200
hidden_size = 20
dropout = 0.00
num_layers = 2
learning_size = 1400
mode = "LSTM"

model = PPLSTM(hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, mode=mode, device=device)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

losses = np.zeros(n_epochs)  # For plotting

# Setup inputs
wave = SimpleSin()
timesteps, _inputs = wave.sample(sample_size=learning_size)
inputs = torch.from_numpy(_inputs[:-1]).float().to(device)
targets = torch.from_numpy(_inputs[1:]).float().to(device)

for epoch in tqdm(range(n_epochs)):
    optimizer.zero_grad()

    outputs = model(inputs, training=True)
    outputs = torch.squeeze(outputs)

    loss = criterion(outputs.view(len(outputs)), targets)
    loss.backward()
    optimizer.step()
    losses[epoch] += loss.item()


plt.plot(range(n_epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

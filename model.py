import torch
from torch import nn
import numpy as np

#make sure batches for rnn are of same length (within the batch)

# load the data
# split pose frames by beats, 

input_seq = torch.from_numpy(input_seq) #music + prev pose
target_seq = torch.Tensor(target_seq) #pose

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if torch.cuda.is_available():
  device = torch.device("cuda")
  print("GPU is available")
else:
  device = torch.device("cpu")
  print("GPU not available, CPU used")

# in future make cuda a requirement

class Model(nn.Module):
  def __init__(self, input_size, output_size, hidden_dim, n_layers):
    super(Model, self).__init__()

    # Defining some parameters
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    #Defining the layers
    # RNN Layer
    self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
    # Fully connected layer
    self.fc = nn.Linear(hidden_dim, output_size)
    
  def forward(self, x):
    batch_size = x.size(0)

    # Initializing hidden state for first input using method defined below
    hidden = self.init_hidden(batch_size)

    # Passing in the input and hidden state into the model and obtaining outputs
    out, hidden = self.rnn(x, hidden)
    
    # Reshaping the outputs such that it can be fit into the fully connected layer
    out = out.contiguous().view(-1, self.hidden_dim)
    out = self.fc(out)
    
    return out, hidden
  
  def init_hidden(self, batch_size):
    # This method generates the first hidden state of zeros which we'll use in the forward pass
    # We'll send the tensor holding the hidden state to the device we specified earlier as well
    hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
    return hidden

# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define hyperparameters
n_epochs = 100
lr=0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training Run
for epoch in range(1, n_epochs + 1):
  optimizer.zero_grad() # Clears existing gradients from previous epoch
  input_seq.to(device)
  output, hidden = model(input_seq)
  loss = criterion(output, target_seq.view(-1).long())
  loss.backward() # Does backpropagation and calculates gradients
  optimizer.step() # Updates the weights accordingly
  
  if epoch%10 == 0:
    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    print("Loss: {:.4f}".format(loss.item()))
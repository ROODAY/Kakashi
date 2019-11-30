import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
  def __init__(self, input_dim, hid_dim, n_layers, dropout):
    super().__init__()
    
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    
    self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, src):
    dropped = self.dropout(src)
    outputs, (hidden, cell) = self.rnn(dropped)

    return hidden, cell

class Decoder(nn.Module):
  def __init__(self, output_dim, hid_dim, n_layers, dropout):
    super().__init__()
    
    self.output_dim = output_dim
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    
    self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout=dropout)
    self.out = nn.Linear(hid_dim, output_dim)
    self.dropout = nn.Dropout2d(dropout)
      
  def forward(self, input, hidden, cell):
    input = input.unsqueeze(0)
    dropped = self.dropout(input)
    seq_len = dropped.shape[0]
    batch_size = dropped.shape[1]
    dropped = dropped.view(seq_len, batch_size, 51) 
    output, (hidden, cell) = self.rnn(dropped, (hidden, cell))
    prediction = self.out(output.squeeze(0))#.reshape((batch_size, 17, 3))
    
    return prediction, hidden, cell

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()
    
    self.encoder = encoder
    self.decoder = decoder
    self.device = device
    
    assert encoder.hid_dim == decoder.hid_dim, \
      "Hidden dimensions of encoder and decoder must be equal!"
    assert encoder.n_layers == decoder.n_layers, \
      "Encoder and decoder must have equal number of layers!"
      
  def forward(self, src, trg, teacher_forcing_ratio=0.5, infer=False):
    if infer:
      assert teacher_forcing_ratio == 0, "Must be zero during inference"
      batch_size = 1
      seq_len = src.shape[0]
    else:
      batch_size = trg.shape[1]
      seq_len = trg.shape[0]
    trg_feature_size = self.decoder.output_dim
    outputs = torch.zeros(seq_len, batch_size, trg_feature_size).to(self.device)
    hidden, cell = self.encoder(src)
    input = trg[0,:]
    
    for t in range(1, seq_len):
      output, hidden, cell = self.decoder(input, hidden, cell)
      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
      input = trg[t] if teacher_force else output
    
    return outputs

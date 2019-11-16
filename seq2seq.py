
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import time
from pathlib import Path

class Encoder(nn.Module):
  def __init__(self, input_dim, hid_dim, n_layers, dropout):
    super().__init__()
    
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    
    self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, src):
    dropped = self.dropout(src)
    #print(dropped)
    #dropped = [src sent len, batch size]
    dropped = dropped.view(1, dropped.shape[0], dropped.shape[1])
    #print('encoder input shape: ', dropped.shape)
    outputs, (hidden, cell) = self.rnn(dropped)
    
    #outputs = [src sent len, batch size, hid dim * n directions]
    #hidden = [n layers * n directions, batch size, hid dim]
    #cell = [n layers * n directions, batch size, hid dim]
    
    #outputs are always from the top hidden layer
    
    return hidden, cell

class Decoder(nn.Module):
  def __init__(self, output_dim, hid_dim, n_layers, dropout):
    super().__init__()
    
    self.output_dim = output_dim
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    
    self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
    self.out = nn.Linear(hid_dim, output_dim)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, input, hidden, cell):
    
    #input = [batch size]
    #hidden = [n layers * n directions, batch size, hid dim]
    #cell = [n layers * n directions, batch size, hid dim]
    
    #n directions in the decoder will both always be 1, therefore:
    #hidden = [n layers, batch size, hid dim]
    #context = [n layers, batch size, hid dim]
    #print('decoder input shape:', input.shape) 
    input = input.unsqueeze(0)
    
    #input = [1, batch size]
    
    dropped = self.dropout(input)
    
    #dropped = [1, batch size]
    #dropped = dropped.view(len(input), len(input[0]), len(input[0][0]))
    dropped = dropped.view(-1).view(1,1,51) 
    #print("decoder dropped shape:", dropped.shape)
    output, (hidden, cell) = self.rnn(dropped, (hidden, cell))
    
    #output = [sent len, batch size, hid dim * n directions]
    #hidden = [n layers * n directions, batch size, hid dim]
    #cell = [n layers * n directions, batch size, hid dim]
    
    #sent len and n directions will always be 1 in the decoder, therefore:
    #output = [1, batch size, hid dim]
    #hidden = [n layers, batch size, hid dim]
    #cell = [n layers, batch size, hid dim]
    prediction = self.out(output.squeeze(0)).reshape((17,3))
    #prediction = [batch size, output dim]
    
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
      
  def forward(self, src, trg, teacher_forcing_ratio = 0.5):
    #src = [src sent len, batch size]
    #trg = [trg sent len, batch size]
    #teacher_forcing_ratio is probability to use teacher forcing
    #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
    #print('target shape:', trg.shape) 
    batch_size = trg.shape[1]
    max_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim
    
    #tensor to store decoder outputs
    outputs = torch.zeros(trg.shape).to(self.device)
    
    #last hidden state of the encoder is used as the initial hidden state of the decoder
    hidden, cell = self.encoder(src)
    #print('encoder output shape:', hidden.shape)
    #first input to the decoder is the <sos> tokens
    input = trg[0,:]
    
    for t in range(1, max_len):
      
      #insert input token embedding, previous hidden and previous cell states
      #receive output tensor (predictions) and new hidden and cell states
      #print('seqseq input:', input)
      output, hidden, cell = self.decoder(input, hidden, cell)
      #print('seqseq output:', output)
      #place predictions in a tensor holding predictions for each token
      outputs[t] = output
      
      #decide if we are going to use teacher forcing or not
      teacher_force = random.random() < teacher_forcing_ratio
      
      #get the highest predicted token from our predictions
      top1 = output.argmax(1) 
      
      #if teacher forcing, use actual next token as next input
      #if not, use predicted token
      input = trg[t] if teacher_force else output
    
    return outputs

def init_weights(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.08, 0.08)

def train(model, iterator, optimizer, criterion, clip):
  model.train()
  
  epoch_loss = 0
  
  for i, batch in enumerate(iterator):
    src = batch['src']
    trg = batch['trg']
    
    optimizer.zero_grad()
   
    print('=> Predicting output...') 
    output = model(src, trg)
    #trg = [trg sent len, batch size]
    #output = [trg sent len, batch size, output dim]
    
    output = output[1:-1]#.view(-1, output.shape[-1])
    output = output.reshape(output.shape[0], model.decoder.output_dim)
    trg = trg[1:-1]#.view(-1)
    trg = trg.reshape(trg.shape[0], model.decoder.output_dim)
    #print("pred: {}, trg: {}".format(output.shape, trg.shape))
    
    #trg = [(trg sent len - 1) * batch size]
    #output = [(trg sent len - 1) * batch size, output dim]
    print('=> Calculating loss...')
    loss = criterion(output, trg)
    print('=> Backpropagating...')
    loss.backward()
    print('=> Clipping gradients...')
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
    optimizer.step()
    
    epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):  
  model.eval()
  
  epoch_loss = 0
  
  with torch.no_grad():  
    for i, batch in enumerate(iterator):
      src = batch['src']
      trg = batch['trg']

      print('=> Predicting output...')
      output = model(src, trg, 0) #turn off teacher forcing

      #trg = [trg sent len, batch size]
      #output = [trg sent len, batch size, output dim]

      output = output[1:-1]#.view(-1, output.shape[-1])
      output = output.reshape(output.shape[0], model.decoder.output_dim)
      trg = trg[1:-1]#.view(-1)
      trg = trg.reshape(trg.shape[0], model.decoder.output_dim)

      #trg = [(trg sent len - 1) * batch size]
      #output = [(trg sent len - 1) * batch size, output dim]
      print('=> Calculating loss...')
      loss = criterion(output, trg)
      
      epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = Path(Path.cwd(), 'data/', 'test')

mfccs = [np.load(path) for path in sorted(list(data_dir.rglob('*.mfcc.npy')))]
max_mfcc_len = max([mfcc.shape[0] for mfcc in mfccs])
mfccs = [np.pad(mfcc, [(max_mfcc_len-len(mfcc), 0), (0,0)]) for mfcc in mfccs]

keypoints = [np.load(path) for path in sorted(list(data_dir.rglob('*.keypoints.npy')))]
max_kp_len = max([kp.shape[0] for kp in keypoints])
keypoints = [np.pad(kp, [(max_kp_len-len(kp), 0), (0,0), (0,0)]) for kp in keypoints]

np.insert(mfccs[0], 0, np.full((20,), -1), axis=0).shape

input_sos = np.full((20,), -0.01)
input_eos = np.full((1,20), 0.01)
output_sos = np.full((1, 17, 3), -0.01)
output_eos = np.full((1, 17, 3), 0.01)

it = [{ 'src': torch.tensor(np.append(np.insert(mfcc, 0, input_sos, axis=0), input_eos, axis=0)).float().to(device), 'trg': torch.tensor(np.append(np.insert(kp, 0, output_sos, axis=0), output_eos, axis=0)).float().to(device)} for mfcc, kp in zip(mfccs, keypoints)]

batch_mfccs = torch.tensor([np.append(np.insert(mfcc, 0, np.zeros((20,)), axis=0), input_eos, axis=0) for mfcc in mfccs]).float()
batch_kps   = torch.tensor([np.append(np.insert(kp, 0, output_sos, axis=0), output_eos, axis=0) for kp in keypoints]).float()
#it = [{'src': batch_mfccs, 'trg': batch_kps}]

#for x in it:
 # print(x) 
 #print('src shape: {}, trg shape: {}'.format(x['src'].shape, x['trg'].shape))
#exit()

INPUT_DIM = 20
OUTPUT_DIM = 51
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)
        
model.apply(init_weights)

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):  
  start_time = time.time()
  
  print('\n=> Training epoch {}\n========'.format(epoch+1))
  train_loss = train(model, it, optimizer, criterion, CLIP)
  print('\n=> Evaluating epoch {}\n========'.format(epoch+1))
  valid_loss = evaluate(model, it, criterion)
  
  end_time = time.time()
  
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  
  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'tut1-model.pt')
  
  print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut1-model.pt'))
test_loss = evaluate(model, it, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

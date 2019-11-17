import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import time
import argparse
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
    dropped = dropped.view(1, dropped.shape[0], dropped.shape[1])
    outputs, (hidden, cell) = self.rnn(dropped)

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
    input = input.unsqueeze(0)
    dropped = self.dropout(input)
    dropped = dropped.view(-1).view(1,1,51) 
    output, (hidden, cell) = self.rnn(dropped, (hidden, cell))
    prediction = self.out(output.squeeze(0)).reshape((17,3))
    
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
    batch_size = trg.shape[1]
    max_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim
    outputs = torch.zeros(trg.shape).to(self.device)
    hidden, cell = self.encoder(src)
    input = trg[0,:]
    
    for t in range(1, max_len):
      output, hidden, cell = self.decoder(input, hidden, cell)
      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
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
    output = model(src, trg, 1)
    output = output[1:-1]
    output = output.reshape(output.shape[0], model.decoder.output_dim)
    trg = trg[1:-1]
    trg = trg.reshape(trg.shape[0], model.decoder.output_dim)

    print('=> Calculating loss...')
    loss = criterion(output, trg)
    print('=> Backpropagating...')
    loss.backward()
    print('=> Clipping gradients...')
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, output_dir):  
  model.eval()
  
  epoch_loss = 0
  with torch.no_grad():  
    for i, batch in enumerate(iterator):
      src = batch['src']
      trg = batch['trg']

      print('=> Predicting output...')
      output = model(src, trg, 0)
      output = output[1:-1]
      np.save(Path(output_dir, '{}.keypoints.npy'.format(str(i+1).zfill(5))), output.cpu().numpy())
      output = output.reshape(output.shape[0], model.decoder.output_dim)
      trg = trg[1:-1]
      trg = trg.reshape(trg.shape[0], model.decoder.output_dim)

      print('=> Calculating loss...')
      loss = criterion(output, trg)
      epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def main(args):
  if args.deterministic:
    SEED = args.seed if args.seed else 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  data_dir = Path(Path.cwd(), 'data/', args.label)

  INPUT_FEATURE = args.input_feature if args.input_feature else 'mfcc'
  inputs = [np.load(path) for path in sorted(list(data_dir.rglob('*.{}.npy'.format(INPUT_FEATURE))))]
  max_inp_len = max([inp.shape[0] for inp in inputs])
  inputs = [np.pad(inp, [(max_inp_len-len(inp), 0), (0,0)]) for inp in inputs]

  keypoints = [np.load(path) for path in sorted(list(data_dir.rglob('*.keypoints.npy')))]
  max_kp_len = max([kp.shape[0] for kp in keypoints])
  keypoints = [np.pad(kp, [(max_kp_len-len(kp), 0), (0,0), (0,0)]) for kp in keypoints]

  input_sos = np.full((20,), -0.01)
  input_eos = np.full((1,20), 0.01)
  output_sos = np.full((1, 17, 3), -0.01)
  output_eos = np.full((1, 17, 3), 0.01)

  it = [{ 'src': torch.tensor(np.append(np.insert(inp, 0, input_sos, axis=0), input_eos, axis=0)).float().to(device), 'trg': torch.tensor(np.append(np.insert(kp, 0, output_sos, axis=0), output_eos, axis=0)).float().to(device)} for inp, kp in zip(inputs, keypoints)]

  INPUT_DIM = 20
  OUTPUT_DIM = 51
  HID_DIM = 512
  N_LAYERS = 2
  ENC_DROPOUT = 0.5
  DEC_DROPOUT = 0.5
  MODEL_NAME = args.model_name if args.model_name else 'kakashi'

  enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
  dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
  model = Seq2Seq(enc, dec, device).to(device)
          
  model.apply(init_weights)

  optimizer = optim.Adam(model.parameters())
  criterion = nn.MSELoss()

  output_dir = Path(Path.cwd(),'out')
  output_dir.mkdir(exist_ok=True)
  run_training = not args.skip_training
  if run_training:
    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):  
      start_time = time.time()
      
      print('=> Training epoch {}\n========'.format(epoch+1))
      train_loss = train(model, it, optimizer, criterion, CLIP)
      print('\n=> Evaluating epoch {}\n========'.format(epoch+1))
      valid_loss = evaluate(model, it, criterion, output_dir)
      
      end_time = time.time()
      
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      
      if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), '{}.pt'.format(MODEL_NAME))
      
      print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}\n')

  model.load_state_dict(torch.load('{}.pt'.format(MODEL_NAME)))
  print('=> Testing model\n========')
  test_loss = evaluate(model, it, criterion, output_dir)
  print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train/infer with Kakashi')
  parser.add_argument('label', type=str,
                      help='Label for the dataset (e.x. Popping)')
  parser.add_argument('--deterministic', action='store_true',
                      help='Train/evaluate deterministically')
  parser.add_argument('--seed', type=int,
                      help='Seed for deterministic run')
  parser.add_argument('--skip_training', action='store_true',
                      help='Skip training phase')
  parser.add_argument('--model_name', type=str,
                      help='Extension of audio files to use for feature extraction (default: .wav)')
  parser.add_argument('--input_feature', type=str,
                      help='Feature set to use for model input')
  args = parser.parse_args()
  main(args)

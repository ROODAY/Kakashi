from models.lstm import Encoder, Decoder, Seq2Seq
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import time
import argparse

def init_weights(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.08, 0.08)

def MSE_Diff(output, target):
  output_diff = output[1:] - output[:-1]
  target_diff = target[1:] - target[:-1]
  return = torch.mean((output_diff - target_diff)**2)

def train(model, iterator, optimizer, criterion, clip):
  model.train()
  
  epoch_loss = 0
  for i, batch in enumerate(tqdm(iterator, desc='Training')):
    src = batch['src']
    trg = batch['trg']
    
    optimizer.zero_grad()
    output = model(src, trg)
    trg = trg.reshape(trg.shape[0], trg.shape[1], model.decoder.output_dim)
    loss = criterion(output, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += loss.item()

  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, output_dir):  
  model.eval()
  
  epoch_loss = 0
  with torch.no_grad():  
    for i, batch in enumerate(tqdm(iterator, desc='Evaluating')):
      src = batch['src']
      trg = batch['trg']

      output = model(src, trg, 0)
      trg = trg.reshape(trg.shape[0], trg.shape[1], model.decoder.output_dim)
      loss = criterion(output, trg)
      epoch_loss += loss.item()

      filename = '{}.keypoints.npy'.format(str(i+1).zfill(5))
      filepath = Path(output_dir, filename)
      seq_len, batch_size, _ = output.shape
      unrolled_features = output.reshape(seq_len, batch_size, 17, 3)
      batch_first = torch.transpose(unrolled_features, 0, 1)
      keypoints = batch_first.cpu().numpy()#batch_first.reshape(batch_size * seq_len, 17, 3).cpu().numpy()
      np.save(filepath, keypoints)
      
  return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def generate_data_splits(inputs, keypoints, device):
  BATCH_SIZE = 10
  SEQ_LEN = 60
  TRAIN_RATIO = 0.7
  VALID_RATIO = 0.2
  TEST_RATIO = 0.1
  # for padding batches
  #max_inp_len = max([inp.shape[0] for inp in inputs])
  #inputs = [np.pad(inp, [(max_inp_len-len(inp), 0), (0,0)]) for inp in inputs]

  #max_kp_len = max([kp.shape[0] for kp in keypoints])
  #keypoints = [np.pad(kp, [(max_kp_len-len(kp), 0), (0,0), (0,0)]) for kp in keypoints]

  zipped = list(zip(inputs, keypoints))
  np.random.shuffle(zipped)
  print('=> Cutting data to SEQ_LEN: {}'.format(SEQ_LEN))
  cut_inputs = []
  cut_keypoints = []
  for inp, kp in zipped:
    groups = len(inp) // SEQ_LEN
    for i in range(1, groups+1):
      cut_inputs.append(inp[(i-1)*SEQ_LEN:i*SEQ_LEN])
      cut_keypoints.append(kp[(i-1)*SEQ_LEN:i*SEQ_LEN])

  print('=> Batching cuts to BATCH_SIZE: {}'.format(BATCH_SIZE))
  batched_inputs = []
  batched_keypoints = []
  batches = len(cut_inputs) // BATCH_SIZE
  for i in range(1, batches+1):
    batched_inputs.append(cut_inputs[(i-1)*BATCH_SIZE:i*BATCH_SIZE])
    batched_keypoints.append(cut_keypoints[(i-1)*BATCH_SIZE:i*BATCH_SIZE])
  
  test_cutoff = round(batches * TEST_RATIO)
  valid_cutoff = round(batches * VALID_RATIO) + test_cutoff
 
  print('=> Creating iterators...')
  train_iterator = [{
    'src': torch.transpose(torch.tensor(batched_inputs[i]), 0, 1).float().to(device),
    'trg': torch.transpose(torch.tensor(batched_keypoints[i]), 0, 1).float().to(device)
  } for i in range(valid_cutoff, batches)]

  valid_iterator = [{
    'src': torch.transpose(torch.tensor(batched_inputs[i]), 0, 1).float().to(device),
    'trg': torch.transpose(torch.tensor(batched_keypoints[i]), 0, 1).float().to(device)
  } for i in range(test_cutoff, valid_cutoff)]
  
  test_iterator = [{
    'src': torch.transpose(torch.tensor(batched_inputs[i]), 0, 1).float().to(device),
    'trg': torch.transpose(torch.tensor(batched_keypoints[i]), 0, 1).float().to(device)
  } for i in range(0, test_cutoff)]

  return (train_iterator, valid_iterator, test_iterator)

def main(args):
  if args.deterministic:
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print('=> Loading Data')
  data_dir = Path(Path.cwd(), 'data/', args.label)

  input_paths = sorted(list(data_dir.rglob('*.{}.npy'.format(args.input_feature))))
  kp_paths = sorted(list(data_dir.rglob('*.keypoints.npy')))
  inputs = [np.load(path) for path in tqdm(input_paths, desc='Loading inputs')]
  keypoints = [np.load(path) for path in tqdm(kp_paths, desc='Loading keypoints')]
  train_iterator, valid_iterator, test_iterator = generate_data_splits(inputs, keypoints, device)

  print('=> Initializing Model')
  INPUT_DIM = 20
  OUTPUT_DIM = 51
  HID_DIM = 512
  N_LAYERS = 2
  ENC_DROPOUT = 0.5
  DEC_DROPOUT = 0.5
  MODEL_NAME = 'kakashi-{}'.format(args.label)

  enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
  dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
  model = Seq2Seq(enc, dec, device).to(device)
          
  model.apply(init_weights)

  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  criterion = MSE_Diff#nn.SmoothL1Loss()#MSELoss()

  output_dir = Path(Path.cwd(),'out/{}'.format(args.label))
  output_dir.mkdir(exist_ok=True, parents=True)
  run_training = not args.skip_training
  if run_training:
    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):  
      start_time = time.time()
      
      print('=> Training epoch {}'.format(epoch+1))
      train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
      print('=> Evaluating epoch {}'.format(epoch+1))
      valid_loss = evaluate(model, valid_iterator, criterion, output_dir)
      
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
  test_loss = evaluate(model, test_iterator, criterion, output_dir)
  print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train Kakashi model')
  parser.add_argument('label', type=str,
                      help='Label for the dataset (e.x. Popping)')
  parser.add_argument('--deterministic', action='store_true',
                      help='Train/evaluate deterministically')
  parser.add_argument('--seed', type=int, default=1234,
                      help='Seed for deterministic run')
  parser.add_argument('--input_feature', type=str, default='mfcc-beat',
                      help='Feature set to use for model input')
  parser.add_argument('--skip_training', action='store_true',
                      help='Skip training phase')
  args = parser.parse_args()
  main(args)

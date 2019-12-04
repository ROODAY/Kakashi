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
import shutil
import yaml
import pickle

def init_weights(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.08, 0.08)

def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / target))

def RPDLoss(output, target):
  return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))

def MAPELoss_Diff(output, target):
  output = output[1:] - output[:-1]
  target = target[1:] - target[:-1]
  return torch.mean(torch.abs((target - output) / target))

def RPDLoss_Diff(output, target):
  output = output[1:] - output[:-1]
  target = target[1:] - target[:-1]
  return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))

def Euclidean_Distance(output, target):
  return torch.sum(torch.sqrt(torch.sum((target-output)**2, dim=3)))

def Velocity_Loss(output, target):
  output = torch.sqrt(torch.sum((output[1:] - output[:-1])**2, dim=3))
  target = torch.sqrt(torch.sum((target[1:] - target[:-1])**2, dim=3))
  return torch.sum(torch.sqrt((target-output)**2))

def Ensemble_Loss(output, target):
  return Euclidean_Distance(output, target) + Velocity_Loss(output, target)

def train(model, iterator, optimizer, criterion, clip, hide_tqdm=False):
  model.train()
  
  epoch_loss = 0
  for i, batch in enumerate(tqdm(iterator, desc='Training', disable=hide_tqdm)):
    src = batch['src']
    trg = batch['trg']
      
    optimizer.zero_grad()
    output = model(src, trg)
    output = output.reshape(trg.shape[0], trg.shape[1], 17, 3)
    loss = criterion(output, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += loss.item()

  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, output_dir, hide_tqdm=False):  
  model.eval()
  
  epoch_loss = 0
  with torch.no_grad():  
    for i, batch in enumerate(tqdm(iterator, desc='Evaluating', disable=hide_tqdm)):
      src = batch['src']
      trg = batch['trg']

      output = model(src, trg, 0)
      output = output.reshape(trg.shape[0], trg.shape[1], 17, 3)
      loss = criterion(output, trg)
      epoch_loss += loss.item()

      filename = '{}.keypoints.npy'.format(str(i+1).zfill(5))
      filepath = Path(output_dir, filename)
      batch_first = torch.transpose(output, 0, 1)
      keypoints = batch_first.cpu().numpy()
      np.save(filepath, keypoints)
      
  return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def generate_data_splits(inputs, keypoints, device, hide_tqdm=False):
  BATCH_SIZE = 10
  SEQ_LEN = 72
  TRAIN_RATIO = 0.7
  VALID_RATIO = 0.2
  TEST_RATIO = 0.1

  print('=> Cutting data to SEQ_LEN: {}'.format(SEQ_LEN))
  cut_inputs = []
  cut_keypoints = []
  for inp, kp in zip(inputs, keypoints):
    groups = len(inp) // SEQ_LEN
    for i in range(1, groups+1):
      cut_inputs.append(inp[(i-1)*SEQ_LEN:i*SEQ_LEN])
      cut_keypoints.append(kp[(i-1)*SEQ_LEN:i*SEQ_LEN])

  print('=> Batching cuts to BATCH_SIZE: {}'.format(BATCH_SIZE))
  zipped = list(zip(cut_inputs, cut_keypoints))
  np.random.shuffle(zipped)
  cut_inputs, cut_keypoints = zip(*zipped)

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
  } for i in tqdm(range(valid_cutoff, batches), desc='Training Iterators', disable=hide_tqdm)]

  valid_iterator = [{
    'src': torch.transpose(torch.tensor(batched_inputs[i]), 0, 1).float().to(device),
    'trg': torch.transpose(torch.tensor(batched_keypoints[i]), 0, 1).float().to(device)
  } for i in tqdm(range(test_cutoff, valid_cutoff), desc='Validation Iterators', disable=hide_tqdm)]
  
  test_iterator = [{
    'src': torch.transpose(torch.tensor(batched_inputs[i]), 0, 1).float().to(device),
    'trg': torch.transpose(torch.tensor(batched_keypoints[i]), 0, 1).float().to(device)
  } for i in tqdm(range(0, test_cutoff), desc='Testing Iterators', disable=hide_tqdm)]

  return (train_iterator, valid_iterator, test_iterator)

def main(args):
  if args.deterministic:
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print('=> Loading Data')
  if args.load_iterators is not None:
    with open(args.load_iterators, 'rb') as f:
        train_iterator, valid_iterator, test_iterator = pickle.load(f)
  else:
    data_dir = Path(Path.cwd(), 'data/', args.label)
    input_paths = sorted(list(data_dir.rglob('*.{}.npy'.format(args.input_feature))))
    kp_paths = sorted(list(data_dir.rglob('*.keypoints.npy')))
    inputs = [np.load(path) for path in tqdm(input_paths, desc='Loading inputs', disable=args.hide_tqdm)]
    keypoints = [np.load(path) for path in tqdm(kp_paths, desc='Loading keypoints', disable=args.hide_tqdm)]
    train_iterator, valid_iterator, test_iterator = generate_data_splits(inputs, keypoints, device, args.hide_tqdm)

    with open(args.save_iterators, 'wb') as f:
      pickle.dump([train_iterator, valid_iterator, test_iterator], f)    

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
  model = Seq2Seq(enc, dec, device)
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
  model.to(device)
  model.apply(init_weights)

  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  criterion = Velocity_Loss

  output_dir = Path(Path.cwd(),'out/{}'.format(args.label))
  if output_dir.exists():
    shutil.rmtree(output_dir)
  output_dir.mkdir(exist_ok=True, parents=True)
  run_training = not args.skip_training
  if run_training:
    N_EPOCHS = 10
    CLIP = 1
    THRESHOLD = 0.01
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):  
      start_time = time.time()
      
      print('=> Training epoch {}'.format(epoch+1))
      train_loss = train(model, train_iterator, optimizer, criterion, CLIP, args.hide_tqdm)
      print('=> Evaluating epoch {}'.format(epoch+1))
      valid_loss = evaluate(model, valid_iterator, criterion, output_dir, args.hide_tqdm)
      
      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      
      print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):7.3f}')
      if valid_loss < best_valid_loss:
        print(f'\tVal. Loss: {valid_loss:.3f} |  Val. PPL: {np.exp(valid_loss):7.3f}')
        print(f'\t New Best Val. Loss\n')
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), '{}.pt'.format(MODEL_NAME))
      else:
        print(f'\tVal. Loss: {valid_loss:.3f} |  Val. PPL: {np.exp(valid_loss):7.3f}\n')

      if valid_loss < THRESHOLD:
        break

  model.load_state_dict(torch.load('{}.pt'.format(MODEL_NAME)))
  print('=> Testing model\n========')
  test_loss = evaluate(model, test_iterator, criterion, output_dir, args.hide_tqdm)
  print(f'| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train Kakashi model')
  parser.add_argument('label', type=str,
                      help='Label for the dataset (e.x. Popping)')
  parser.add_argument('--deterministic', action='store_true',
                      help='Train/evaluate deterministically')
  parser.add_argument('--seed', type=int, default=1234,
                      help='Seed for deterministic run')
  parser.add_argument('--input_feature', type=str, default='mfcc-frame',
                      help='Feature set to use for model input')
  parser.add_argument('--load_iterators', type=str,
                      help='Load iterators directly from file instead of generating')
  parser.add_argument('--save_iterators', type=str, default='checkpoint.pkl',
                      help='Save iterators into file')
  parser.add_argument('--skip_training', action='store_true',
                      help='Skip training phase')
  parser.add_argument('--hide_tqdm', action='store_true',
                      help='Hide tqdm output')
  args = parser.parse_args()
  main(args)

from models.lstm import Encoder, Decoder, Seq2Seq
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
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

# Initialize model parameters to a uniform random distribution
def init_weights(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.08, 0.08)

# Summed Euclidean Distances
def Euclidean_Distance(output, target):
  return torch.sum(torch.sqrt(torch.sum((target-output)**2, dim=3)))

# Mean Absolute Percentage Error
def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / target))

# Relative Percent Difference
def RPDLoss(output, target):
  return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))

# MAPE on the pose velocities
def MAPELoss_Velocity(output, target):
  output = torch.sqrt(torch.sum((output[1:] - output[:-1])**2, dim=3))
  target = torch.sqrt(torch.sum((target[1:] - target[:-1])**2, dim=3))
  return torch.mean(torch.abs((target - output) / target))

# RPD on the pose velocities
def RPDLoss_Velocity(output, target):
  output = torch.sqrt(torch.sum((output[1:] - output[:-1])**2, dim=3))
  target = torch.sqrt(torch.sum((target[1:] - target[:-1])**2, dim=3))
  return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))

# SRSE on the pose velocities
def SRSE_Velocity(output, target):
  output = torch.sqrt(torch.sum((output[1:] - output[:-1])**2, dim=3))
  target = torch.sqrt(torch.sum((target[1:] - target[:-1])**2, dim=3))
  return torch.sum(torch.sqrt((target-output)**2))

# Attempt to train on both pose coordinates and velocities
def Ensemble_Loss(output, target):
  return Euclidean_Distance(output, target) + SRSE_Velocity(output, target)

def train(model, iterator, optimizer, criterion, clip, hide_tqdm=False):
  model.train()
  
  epoch_loss = 0
  for i, batch in enumerate(tqdm(iterator, desc='Training', disable=hide_tqdm)):
    src = batch['src']
    trg = batch['trg']
      
    optimizer.zero_grad()
    output = model(src, trg)

    # Reshape output to match target shape and calculate loss
    output = output.reshape(trg.shape[0], trg.shape[1], 17, 3)
    loss = criterion(output, trg)

    # Stop execution if NaN is encountered
    if torch.isnan(loss).any():
      print('=> ERROR: NaN in loss for batch {}'.format(i))
      exit()

    # Backpropagate and update weights
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

      # Reshape output to match target shape and calculate loss
      output = output.reshape(trg.shape[0], trg.shape[1], 17, 3)
      loss = criterion(output, trg)

      # Stop execution if NaN is encountered
      if torch.isnan(loss).any():
        print('=> ERROR: NaN in loss for batch {}'.format(i))
        exit()
      epoch_loss += loss.item()

      # Save keypoints from evaluation for analysis later
      filename = '{}.keypoints.npy'.format(str(i+1).zfill(5))
      filepath = Path(output_dir, filename)
      batch_first = torch.transpose(output, 0, 1)
      keypoints = batch_first.cpu().numpy()
      np.save(filepath, keypoints)
      
  return epoch_loss / len(iterator)

# Helper function to calculate length of epoch
def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

# Helper function to batch data
def generate_data_splits(inputs, keypoints, device, config, hide_tqdm=False):
  # Load values from config
  BATCH_SIZE = config['BATCH_SIZE']
  SEQ_LEN = config['SEQ_LEN']
  TRAIN_RATIO = config['TRAIN_RATIO']
  VALID_RATIO = config['VALID_RATIO']
  TEST_RATIO = config['TEST_RATIO']

  # Cut audio and pose inputs to desired sequence length (some info is discarded)
  print('=> Cutting data to SEQ_LEN: {}'.format(SEQ_LEN))
  cut_inputs = []
  cut_keypoints = []
  for inp, kp in zip(inputs, keypoints):
    groups = len(inp) // SEQ_LEN
    for i in range(1, groups+1):
      cut_inputs.append(inp[(i-1)*SEQ_LEN:i*SEQ_LEN])
      cut_keypoints.append(kp[(i-1)*SEQ_LEN:i*SEQ_LEN])

  print('=> Batching cuts to BATCH_SIZE: {}'.format(BATCH_SIZE))
  # Shuffle data before batching
  zipped = list(zip(cut_inputs, cut_keypoints))
  np.random.shuffle(zipped)
  cut_inputs, cut_keypoints = zip(*zipped)

  # Batch data to desired batch size (some info is discarded)
  batched_inputs = []
  batched_keypoints = []
  batches = len(cut_inputs) // BATCH_SIZE
  for i in range(1, batches+1):
    batched_inputs.append(cut_inputs[(i-1)*BATCH_SIZE:i*BATCH_SIZE])
    batched_keypoints.append(cut_keypoints[(i-1)*BATCH_SIZE:i*BATCH_SIZE])
  
  print('=> Creating iterators...')
  test_cutoff = round(batches * TEST_RATIO)
  valid_cutoff = round(batches * VALID_RATIO) + test_cutoff

  # Put data on tensors on the target device
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
  with open(args.config) as f:
    config = yaml.full_load(f)

  if args.deterministic:
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print('=> Loading Data')
  # If iterators were passed, load instead of regenerating
  if args.load_iterators is not None:
    with open(args.load_iterators, 'rb') as f:
        train_iterator, valid_iterator, test_iterator = pickle.load(f)
  else:
    data_dir = Path(Path.cwd(), 'data/', args.label)
    input_paths = sorted(list(data_dir.rglob('*.{}.npy'.format(args.input_feature))))
    kp_paths = sorted(list(data_dir.rglob('*.keypoints.npy')))
    inputs = [np.load(path) for path in tqdm(input_paths, desc='Loading inputs', disable=args.hide_tqdm)]
    keypoints = [np.load(path) for path in tqdm(kp_paths, desc='Loading keypoints', disable=args.hide_tqdm)]
    train_iterator, valid_iterator, test_iterator = generate_data_splits(inputs, keypoints, device, config['data-split'], args.hide_tqdm)

    # Save iterators for later use
    Path(args.save_iterators).parent.mkdir(exist_ok=True, parents=True)
    with open(args.save_iterators, 'wb') as f:
      pickle.dump([train_iterator, valid_iterator, test_iterator], f)    

  print('=> Initializing Model')
  INPUT_DIM = config['model']['INPUT_DIM']
  OUTPUT_DIM = config['model']['OUTPUT_DIM']
  HID_DIM = config['model']['HID_DIM']
  N_LAYERS = config['model']['N_LAYERS']
  ENC_DROPOUT = config['model']['ENC_DROPOUT']
  DEC_DROPOUT = config['model']['DEC_DROPOUT']

  enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
  dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
  model = Seq2Seq(enc, dec, device)
  model.to(device)
  model.apply(init_weights)

  optimizer = optim.SGD(model.parameters(), lr=config['model']['LEARNING_RATE'], momentum=config['model']['MOMENTUM'])

  if config['model']['LOSS_FUNCTION'] == 'MAPELoss':
    criterion = MAPELoss
  elif config['model']['LOSS_FUNCTION'] == 'RPDLoss':
    criterion = RPDLoss
  elif config['model']['LOSS_FUNCTION'] == 'MAPELoss_Velocity':
    criterion = MAPELoss_Velocity
  elif config['model']['LOSS_FUNCTION'] == 'RPDLoss_Velocity':
    criterion = RPDLoss_Velocity
  elif config['model']['LOSS_FUNCTION'] == 'Euclidean_Distance':
    criterion = Euclidean_Distance
  elif config['model']['LOSS_FUNCTION'] == 'SRSE_Velocity':
    criterion = SRSE_Velocity
  elif config['model']['LOSS_FUNCTION'] == 'Ensemble_Loss':
    criterion = Ensemble_Loss
  elif config['model']['LOSS_FUNCTION'] == 'L1Loss':
    criterion = nn.L1Loss()
  elif config['model']['LOSS_FUNCTION'] == 'MSELoss':
    criterion = nn.MSELoss()
  elif config['model']['LOSS_FUNCTION'] == 'SmoothL1Loss':
    criterion = nn.SmoothL1Loss()
  else:
    print('=> ERROR: Loss function not defined')
    exit()

  # Setup output directory, clear if necessary
  output_dir = Path(Path.cwd(),'out/{}'.format(args.model_name))
  if output_dir.exists():
    shutil.rmtree(output_dir)
  output_dir.mkdir(exist_ok=True, parents=True)

  run_training = not args.skip_training
  if run_training:
    N_EPOCHS = config['training']['N_EPOCHS']
    CLIP = config['training']['CLIP']
    THRESHOLD = config['training']['THRESHOLD']
    best_valid_loss = float('inf')

    train_losses = []
    valid_losses = []
    epochs = [i+1 for i in range(N_EPOCHS)]

    for epoch in epochs:  
      start_time = time.time()
      
      print('=> Training epoch {}'.format(epoch))
      train_loss = train(model, train_iterator, optimizer, criterion, CLIP, args.hide_tqdm)
      print('=> Evaluating epoch {}'.format(epoch))
      valid_loss = evaluate(model, valid_iterator, criterion, output_dir, args.hide_tqdm)

      train_losses.append(train_loss)
      valid_losses.append(valid_loss)
      
      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      
      # If validation loss is best seen so far, save the current model
      print(f'Epoch: {epoch:03} | Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f}')
      if valid_loss < best_valid_loss:
        print(f'\t Val. Loss: {valid_loss:.3f}')
        print(f'\t New Best Val. Loss\n')
        best_valid_loss = valid_loss
        model_path = Path(Path.cwd(), 'pre/{}.best_valid.pt'.format(args.model_name))
        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), model_path)
      else:
        print(f'\t Val. Loss: {valid_loss:.3f}\n')

      if valid_loss < THRESHOLD:
        print('=> Valid Loss under THRESHOLD: {}'.format(THRESHOLD))
        break

    # Save fully trained model to compare with best valid loss model
    model_path = Path(Path.cwd(), 'pre/{}.trained.pt'.format(args.model_name))
    model_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_path)

    # Save plot of training vs validation loss
    plot_path = Path(Path.cwd(), 'experiments/{}.png'.format(args.model_name))
    plt.plot(epochs, train_losses, color='blue')
    plt.plot(epochs, valid_losses, color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(plot_path, bbox_inches='tight')

  model.load_state_dict(torch.load(model_path))
  print('=> Testing model')
  test_loss = evaluate(model, test_iterator, criterion, output_dir, args.hide_tqdm)
  print(f'| Test Loss: {test_loss:.3f} |')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train Kakashi model')
  parser.add_argument('label', type=str,
                      help='Label for the dataset (e.x. Popping)')
  parser.add_argument('--deterministic', action='store_true',
                      help='Train/evaluate deterministically')
  parser.add_argument('--seed', type=int, default=1234,
                      help='Seed for deterministic run')
  parser.add_argument('--config', type=str, default='config/default.yaml',
                      help='Config file to load')
  parser.add_argument('--input_feature', type=str, default='mfcc-frame',
                      help='Feature set to use for model input')
  parser.add_argument('--model_name', type=str, default='kakashi',
                      help='Name for saved model file')
  parser.add_argument('--load_iterators', type=str,
                      help='Load iterators directly from file instead of generating')
  parser.add_argument('--save_iterators', type=str, default='its/checkpoint.pkl',
                      help='Save iterators into file')
  parser.add_argument('--skip_training', action='store_true',
                      help='Skip training phase')
  parser.add_argument('--hide_tqdm', action='store_true',
                      help='Hide tqdm output')
  args = parser.parse_args()
  main(args)

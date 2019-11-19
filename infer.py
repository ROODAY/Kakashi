from models.seq2seq import Seq2Seq
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np
import random
import math
import time
import argparse

def main(args):

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
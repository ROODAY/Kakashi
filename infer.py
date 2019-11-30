from models.seq2seq import Seq2Seq
from pathlib import Path
import torch
import numpy as np
import argparse
import librosa

def main(args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
  model_path = Path(args.model_path)
  model.load_state_dict(torch.load(model_path))

  print('=> Processing Input')
  input_path = Path(args.input_path)
  audio_data, sample_rate = librosa.load(input_path, res_type='kaiser_fast')
  frames = round(librosa.get_duration(y=audio_data, sr=sample_rate) * 30)
  grouped_audio = np.array_split(audio_data, frames)
  features = np.array([np.mean(librosa.feature.mfcc(y=group).T,axis=0) for group in grouped_audio])
  features = features.reshape(features.shape[0], 1, features.shape[1])

  print('=> Running Model')
  output = model(feature, None, 0, True)

  output_path = Path(Path.cwd(), 'out/infer/{}.keypoints.npy'.format(input_path.stem))
  print('=> Saving output to {}'.format(output_path))
  seq_len, batch_size, _ = output.shape
  unrolled_features = output.reshape(seq_len, batch_size, 17, 3)
  batch_first = torch.transpose(unrolled_features, 0, 1)
  keypoints = batch_first.cpu().numpy()
  np.save(output_path, keypoints)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Infer choreography from audio file')
  parser.add_argument('model_path', type=str,
                      help='Path of model file to use')
  parser.add_argument('input_path', type=str,
                      help='Path of input file to use')
  args = parser.parse_args()
  main(args)
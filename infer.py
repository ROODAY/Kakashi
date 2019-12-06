from models.lstm import Encoder, Decoder, Seq2Seq
from pathlib import Path
import torch
import numpy as np
import argparse
import librosa
import random
import os
import subprocess
import yaml

def main(args):
  with open(args.config) as f:
    config = yaml.full_load(f)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print('=> Initializing Model')
  INPUT_DIM = config['model']['INPUT_DIM']
  OUTPUT_DIM = config['model']['OUTPUT_DIM']
  HID_DIM = config['model']['HID_DIM']
  N_LAYERS = config['model']['N_LAYERS']
  ENC_DROPOUT = config['model']['ENC_DROPOUT']
  DEC_DROPOUT = config['model']['DEC_DROPOUT']
  FRAME_RATE = config['inference']['FRAME_RATE']
  SEQ_LEN = config['inference']['SEQ_LEN']

  enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
  dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
  model = Seq2Seq(enc, dec, device).to(device)
  model_path = Path(args.model_path)
  model.load_state_dict(torch.load(model_path))

  print('=> Processing Input')
  # Load input, and if necessary extract features
  input_path = Path(args.input_path)
  if args.raw_features:
    features = np.load(input_path)
  else:
    audio_data, sample_rate = librosa.load(input_path, res_type='kaiser_fast')
    frames = round(librosa.get_duration(y=audio_data, sr=sample_rate) * FRAME_RATE)
    grouped_audio = np.array_split(audio_data, frames)
    features = [np.mean(librosa.feature.mfcc(y=group).T,axis=0) for group in grouped_audio]

  # Convert features to float tensor, reshape to (seq_len, batch_size=1, feature_len), and push to device
  features = torch.tensor(features).float()
  features = features.reshape(features.shape[0], 1, features.shape[1]).to(device)

  print('=> Selecting Seed Pose')
  # Pick a pose to start the dance
  all_poses = list(Path(Path.cwd(), 'data/{}'.format(args.seed_label)).rglob('*.keypoints.npy'))
  seed_pose = torch.tensor(np.load(random.choice(all_poses))[:1])

  with torch.no_grad():
    print('=> Running Model')
    output = model(features, None, 0, True, seed_pose)

    output_dir = Path(Path.cwd(), 'out/infer')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = Path(output_dir, '{}.keypoints.npy'.format(input_path.stem))

    print('=> Saving output to {}'.format(output_path))
    seq_len, _, _ = output.shape
    # Remove batch dimension before saving so we can render if desired
    unrolled_features = output.reshape(seq_len, 17, 3)
    keypoints = unrolled_features.cpu().numpy()
    np.save(output_path, keypoints)

  if args.render:
    print('=> Calling Renderer')
    # Use custom VideoPose script to render keypoints
    os.chdir(os.environ['VIDEOPOSE'])
    if args.render_name is not None:
      video_path = Path(output_dir, args.render_name)
    else:
      video_path = Path(output_dir, '{}.mp4'.format(input_path.stem))
    command = 'python3 animate.py --viz-input {} --viz-output {}'.format(output_path, video_path)
    subprocess.call(command, shell=True)

    if not args.raw_features:
      print('=> Muxing')
      # Mux rendered video with original audio
      os.chdir(os.environ['KAKASHI'])
      tmp_path = Path(output_dir, '{}.tmp.mp4'.format(video_path.stem))
      command = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental {}'.format(video_path, input_path, tmp_path)
      subprocess.call(command, shell=True)
      tmp_path.replace(video_path)

    print('=> Final output stored at {}'.format(video_path))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Infer choreography from audio file')
  parser.add_argument('model_path', type=str,
                      help='Path of model file to use')
  parser.add_argument('input_path', type=str,
                      help='Path of input file to use')
  parser.add_argument('--config', type=str, default='config/default.yaml',
                      help='Config file to load')
  parser.add_argument('--seed_label', type=str, default='wod',
                      help='Dataset label to grab seed frame from')
  parser.add_argument('--render', action='store_true',
                      help='Render saved keypoints')
  parser.add_argument('--render_name', type=str,
                      help='Name for rendered output')
  parser.add_argument('--raw_features', action='store_true',
                      help='Input file being passed is already a set of features')
  args = parser.parse_args()
  main(args)

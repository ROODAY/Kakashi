from pathlib import Path
from tqdm import tqdm
from itertools import islice
import shutil
import numpy as np
import librosa
import subprocess
import os
import argparse

def window(seq, n=2):
  it = iter(seq)
  result = tuple(islice(it, n))
  if len(result) == n:
    yield result
  for elem in it:
    result = result[1:] + (elem,)
    yield result

def main(args):
  videpose_dataset_name = args.videpose_dataset_name

  # make sure video directory exists
  download_dir = Path(Path.cwd(), 'data/', args.label)
  if not download_dir.exists():
    raise NotADirectoryError('{} does not exist!'.format(str(download_dir)))

  video_paths = sorted(list(Path(download_dir).rglob('*.mp4')))

  if not args.skip_detect_pose:
    # copy videos to tmp directory for pose estimation
    print('=> Copying videos to tmp directory for pose estimation')
    tmp_input_dir = Path(Path.cwd(), 'tmp_in/')
    tmp_input_dir.mkdir(exist_ok=True) 
    tmp_output_dir = Path(Path.cwd(), 'tmp_out/')
    tmp_output_dir.mkdir(exist_ok=True) 
    for video_path in video_paths:
      shutil.copy(str(video_path), str(Path(tmp_input_dir, video_path.name)))

    # run 2D detections (SLOW/BOTTLENECK)
    os.chdir(os.environ['DETECTRON'])
    print('=> Running 2D pose detection (this is going to take a while...)')
    command = 'python3 tools/infer_video.py --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir {} --image-ext mp4 --wts models/model_final.pkl {}'.format(tmp_output_dir, tmp_input_dir)
    subprocess.call(command, shell=True)

    # prepare 2D dataset
    os.chdir(Path(os.environ['VIDEOPOSE'], 'data'))
    print('=> Preparing 2D keypoint dataset')
    command = 'python3 prepare_data_2d_custom.py -i {} -o {}'.format(tmp_output_dir, videpose_dataset_name)
    subprocess.call(command, shell=True)
    dataset_path = Path(Path.cwd(), 'data_2d_custom_{}.npz'.format(videpose_dataset_name))

    # get 3D detections
    os.chdir(os.environ['VIDEOPOSE'])
    print('=> Running 3D pose detection')
    for video_path in video_paths:
      print('=> Processing: {}'.format(video_path))
      command = 'python3 run.py -d custom -k {} -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject {} --viz-action custom --viz-camera 0 --viz-video {} --viz-export {} --viz-size 6'.format(videpose_dataset_name, video_path.name, str(video_path), Path(video_path.parent, video_path.stem + '.keypoints.npy'))
      subprocess.call(command, shell=True)

    if not args.save_tmp:
      # delete tmp folders
      print('=> Deleting tmp folders')
      shutil.rmtree(tmp_input_dir)
      shutil.rmtree(tmp_output_dir)

  # extract audio features
  os.chdir(os.environ['KAKASHI'])
  if not args.skip_extract:
    for video_path in video_paths:
      print('=> Extracting Audio for {}'.format(video_path.name))
      audio_path = Path(video_path.parent, '{}.wav'.format(video_path.stem))
      command = 'ffmpeg -hide_banner -loglevel panic -y -i {} -ab 160k -ac 2 -ar 44100 -vn {}'.format(video_path, audio_path)
      subprocess.call(command, shell=True)

  audio_ext = args.audio_ext
  audio_feature = args.audio_feature
  audio_paths = list(Path(download_dir).rglob('*.{}'.format(audio_ext)))
  audio_paths.sort()
  for audio_path in audio_paths:
    print('=> Processing features for {}'.format(audio_path.name))
    audio_data, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')

    feature_type, feature_split = audio_feature.split('-')
    if feature_split == 'beat':
      print('=> Group audio data by beat')
      tempo, beat_frames = librosa.beat.beat_track(audio_data, sr=sample_rate)
      indices = list(window(beat_frames))
      grouped_audio = [audio_data[start:stop] for (start, stop) in indices]
    elif feature_split == 'time':
      print('=> Group audio data by time')
      interval = args.time_interval
      buckets = round(librosa.get_duration(y=audio_data, sr=sample_rate) / interval)
      grouped_audio = np.array_split(audio_data, buckets)
    elif feature_split == 'pose':
      print('=> Group audio data by pose')
      pose_path = Path(audio_path.parent, '{}.keypoints.npy'.format(audio_path.stem))
      poses = np.load(pose_path)
      grouped_audio = np.array_split(audio_data, len(poses))
    else:
      raise AssertionError('{} is not valid!'.format(audio_feature))
      
    if feature_type == 'mfcc':
      print('=> Extract MFCC features')
      features = [np.mean(librosa.feature.mfcc(y=group).T,axis=0) for group in tqdm(grouped_audio)]
    else:
      raise AssertionError('{} is not valid!'.format(audio_feature))

    audio_feature = '{}_{}'.format(audio_feature, interval) if feature_split == 'time' else audio_feature
    np.save(Path(audio_path.parent, '{}.{}.npy'.format(audio_path.stem, audio_feature)), features)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generates dataset from video files')
  parser.add_argument('label', type=str,
                      help='Label for the dataset (e.x. Popping)')
  parser.add_argument('--videpose_dataset_name', type=str, default='kakashi',
                      help='Name for VideoPose custom dataset (default: kakashi)')
  parser.add_argument('--audio_ext', type=str, default='wav',
                      help='Extension of audio files to use for feature extraction (default: .wav)')
  parser.add_argument('--audio_feature', type=str, default='mfcc-beat',
                      help='Type of audio feature to extract (default: mfcc per beat)')
  parser.add_argument('--time_interval', type=int, default=5,
                      help='Length of interval in seconds for mfcc-time (default: 5)')
  parser.add_argument('--save_tmp', action='store_true',
                      help='Save tmp folders instead of deleting')
  parser.add_argument('--skip_detect_pose', action='store_true',
                      help='Skip running pose detection')
  parser.add_argument('--skip_extract', action='store_true',
                      help='Skip extracting of audio from video files (if audio was downloaded separately)')
  args = parser.parse_args()
  main(args)

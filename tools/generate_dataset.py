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
  videpose_dataset_name = args.videpose_dataset_name if args.videpose_dataset_name else 'kakashi'

  # make sure video directory exists
  download_dir = Path(Path.cwd(), 'data/', args.label)
  if not download_dir.exists():
    raise NotADirectoryError('{} does not exist!'.format(str(download_dir)))

  video_paths = list(Path(download_dir).rglob('*.mp4'))
  video_paths.sort()

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

    # delete tmp folders
    print('=> Deleting tmp folders')
    shutil.rmtree(tmp_input_dir)
    shutil.rmtree(tmp_output_dir)

  # extract audio features
  os.chdir(os.environ['KAKASHI'])
  if not args.skip_extract:
    print('=> Extracting audio features')
    for video_path in video_paths:
      print('=> Extracting Audio for {}'.format(video_path.name))
      audio_path = Path(video_path.parent, '{}.wav'.format(video_path.stem))
      command = 'ffmpeg -hide_banner -loglevel panic -y -i {} -ab 160k -ac 2 -ar 44100 -vn {}'.format(video_path, audio_path)
      subprocess.call(command, shell=True)

  audio_ext = args.audio_ext if args.audio_ext else 'wav'
  audio_paths = list(Path(download_dir).rglob('*.{}'.format(audio_ext)))
  audio_paths.sort()
  for audio_path in audio_paths:
    print('=> Processing features for {}'.format(audio_path.name))
    audio_data, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')

    print('=> Group audio data by beat')
    tempo, beat_frames = librosa.beat.beat_track(audio_data, sr=sample_rate)
    indices = list(window(beat_frames))
    audio_by_beat = [audio_data[start:stop] for (start, stop) in indices]

    print('=> Get MFCCs per beat')
    mfccs = [np.mean(librosa.feature.mfcc(y=beat, n_mfcc=17).T,axis=0) for beat in tqdm(audio_by_beat)]
    np.save(Path(audio_path.parent, '{}.mfcc.npy'.format(audio_path.stem)), mfccs)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generates dataset from video files')
  parser.add_argument('label', type=str,
                      help='Label for the dataset (e.x. Popping)')
  parser.add_argument('--videpose_dataset_name', type=str,
                      help='Name for VideoPose custom dataset (default: kakashi)')
  parser.add_argument('--skip_extract', action='store_true',
                      help='Skip extracting of audio from video files (if audio was downloaded separately)')
  parser.add_argument('--audio_ext', type=str,
                      help='Extension of audio files to use for feature extraction (default: .wav)')
  parser.add_argument('--skip_detect_pose', action='store_true',
                      help='Skip running pose detection')
  args = parser.parse_args()
  main(args)

# keypoints = np.load('keypoints.npz.npy', allow_pickle=True) same for mfcc

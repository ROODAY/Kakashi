#from __future__ import unicode_literals
from pathlib import Path
from tqdm import tqdm
import shutil
import numpy as np
import librosa
import subprocess
import os
import argparse

def main(args):
  # make sure video directory exists
  download_dir = Path(Path.cwd(), 'data/', args.label)
  if not download_dir.exists():
    raise NotADirectoryError('{} does not exist!'.format(str(download_dir)))

  # copy videos to tmp directory for pose estimation
  print('=> Copying videos to tmp directory for pose estimation')
  video_paths = Path(download_dir).rglob('*.mp4')
  tmp_input_dir = Path(Path.cwd(), 'tmp_in/')
  tmp_input_dir.mkdir(exist_ok=True) 
  tmp_output_dir = Path(Path.cwd(), 'tmp_out/')
  tmp_output_dir.mkdir(exist_ok=True) 
  for video_path in video_paths:
    shutil.copy(str(video_path), str(Path(tmp_input_dir, video_path.name)))

  # run 2D detections (SLOW/BOTTLENECK)
  os.chdir(os.environ['DETECTRON'])
  print('=> Running 2D pose detection (this is going to take a while...)')
  command = 'python3 tools/infer_video.py --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir {} --image-ext mp4 --wts https://dl.fbaipublicfiles.com/detectron/37698009/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml.08_45_57.YkrJgP6O/output/train/keypoints_coco_2014_train:keypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl {}'.format(tmp_output_dir, tmp_input_dir)
  subprocess.call(command, shell=True)

  # prepare 2D dataset
  os.chdir(os.environ['VIDEOPOSE'])
  print('=> Preparing 2D keypoint dataset')
  command = 'python3 prepare_data_2d_custom.py -i {} -o myvideos'.format(tmp_output_dir)
  dataset_path = Path(Path.cwd(), 'data/data_2d_custom_myvideos.npz')

  # get 3D detections
  print('=> Running 3D pose detection')
  for video_path in video_paths:
    command = 'python3 run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject {} --viz-action custom --viz-camera 0 --viz-video {} --viz-export {} --viz-size 6'.format(video_path.name, video_path, Path(video_path.parent, video_path.stem + '.keypoints.npy'))

  # delete tmp folders
  print('=> Deleting tmp folders')
  shutil.rmtree(tmp_input_dir)
  shutil.rmtree(tmp_output_dir)

  # extract audio features
  os.chdir(os.environ['KAKASHI'])
  print('=> Extracting audio features')
  extract_audio = not args.skip_extract
  if extract_audio:
    for video_path in video_paths:
      print('=> Extracting Audio for {}'.format(video_path.name))
      audio_path = Path(video_path.parent, '{}.wav'.format(video_path.stem))
      command = 'ffmpeg -hide_banner -loglevel panic -y -i {} -ab 160k -ac 2 -ar 44100 -vn {}'.format(video_path, audio_path)
      subprocess.call(command, shell=True)

      # break out this into another step after audio extraction, add arg for audio ext and do a glob
      print('=> Loading Audio into librosa')
      audio_data, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')

      print('=> Group audio data by frame')
      audio_by_frame = np.array_split(audio_data, num_frames) # fix this line to get number of frames better

      print('=> Get MFCCs per frame')
      mfccs = [np.mean(librosa.feature.mfcc(y=frame).T,axis=0) for frame in tqdm(audio_by_frame)]

      print('=> MFCC Length: {}'.format(len(mfccs)))
      print('=> Length of one MFCC: {}'.format(len(mfccs[0])))
      np.savetxt(Path(video_path.parent, '{}.audio.gz'.format(video_path.stem)), mfccs)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generates dataset from video files')
  parser.add_argument('label', type=str,
                      help='Label for the dataset (e.x. Popping)')
  parser.add_argument('--skip_extract', action='store_true',
                      help='Skip extracting of audio from video files (if audio was downloaded separately)')
  args = parser.parse_args()
  main(args)

# keypoints = np.load('keypoints.npz.npy', allow_pickle=True) same for mfcc

from __future__ import unicode_literals
from pathlib import Path
from tqdm import tqdm
import shutil
import youtube_dl
import numpy as np
import librosa
import subprocess
import os
import argparse

# for cleanliness, at the end separate into functions and make a main/init function

parser = argparse.ArgumentParser(description='Generates dataset for Kakashi')
parser.add_argument('--playlist_url', type=str,
                    help='The YouTube playlist to get videos from (default is the Kakashi Raw Playlist)')
parser.add_argument('--separate_AV', action='store_true',
                    help='Prevent muxing of Audio/Video into one stream (used when videos don\'t need to be cut)')

args = parser.parse_args()
playlist_url = args.playlist_url if args.playlist_url != None else 'https://www.youtube.com/watch?v=htLt1UvEMLE&list=PL8Kqpe3GkRgGNlb-RYgT8oBmPLuajZu4X'
separate_AV = args.separate_AV
extract_audio = not separate_AV

# prepare path for data
label = input('Label for dataset: ')
download_dir = Path(Path.cwd(), 'data/', label)
Path(download_dir).mkdir(parents=True, exist_ok=True) 

# download playlist
name_scheme = '%(id)s/%(id)s.%(ext)s' if separate_AV else '%(autonumber)s/%(autonumber)s.%(ext)s'
final_format = 'bestvideo[ext=mp4],bestaudio[ext=m4a]' if separate_AV else 'bestvideo[ext=mp4]+bestaudio[ext=m4a]'
ydl_opts = {
  'outtmpl': '{}/{}'.format(download_dir, name_scheme),
  'restrictfilenames': True,
  'writeinfojson': False,
  'postprocessor-args': '-strict -2',
  'format': final_format
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([playlist_url])

# if video/audio downloaded separately and needs renaming
if separate_AV:
  print('=> Renaming files')
  count = 0
  dirs = Path(download_dir).glob('*/')
  for folder in dirs:
    for _,_,files in os.walk(folder): # replace this with Pathlib version, needs a rewrite
      for file in files:
        name = file.split('.')
        name[0] = str(count).zfill(5)
        new_name = '.'.join(name)
        old_path = Path(folder, file)
        new_path = Path(folder, new_name)
        old_path.rename(new_path)
    folder.rename(Path(folder.parent, str(count).zfill(5)))
    count += 1

# copy videos to tmp directory for pose estimation
video_paths = Path(download_dir).rglob('*.mp4')
tmp_input_dir = Path(Path.cwd(), 'tmp_in/')
tmp_output_dir = Path(Path.cwd(), 'tmp_out/')
for video_path in video_paths:
  shutil.copy(str(video_path), str(Path(tmp_input_dir, video_path.name)))

# run 2D detections (SLOW/BOTTLENECK)
os.chdir(os.environ['DETECTRON'])
command = 'python3 tools/infer_video.py --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir {} --image-ext mp4 --wts https://dl.fbaipublicfiles.com/detectron/37698009/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml.08_45_57.YkrJgP6O/output/train/keypoints_coco_2014_train:keypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl {}'.format(tmp_output_dir, tmp_input_dir)
subprocess.call(command, shell=True)

# prepare 2D dataset
os.chdir(os.environ['VIDEOPOSE'])
command = 'python3 prepare_data_2d_custom.py -i {} -o myvideos'.format(tmp_output_dir)
dataset_path = Path(Path.cwd(), 'data/data_2d_custom_myvideos.npz')

# get 3D detections
for video_path in video_paths:
  command = 'python3 run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject {} --viz-action custom --viz-camera 0 --viz-video {} --viz-export {} --viz-size 6'.format(video_path.name, video_path, Path(video_path.parent, video_path.stem + '.keypoints.npy'))

# extract audio features
os.chdir(os.environ['KAKASHI'])
if extract_audio:
  for video_path in video_paths:
    print('\n=> {}'.format(video_path))

    print('=> Extracting Audio')
    audio_path = Path(video_path.parent, '{}.wav'.format(video_path.stem))
    command = 'ffmpeg -hide_banner -loglevel panic -y -i {} -ab 160k -ac 2 -ar 44100 -vn {}'.format(video_path, audio_path)
    subprocess.call(command, shell=True)

    print('=> Loading Audio into librosa')
    audio_data, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')

    print('=> Group audio data by frame')
    audio_by_frame = np.array_split(audio_data, num_frames) # fix this line to get number of frames better

    print('=> Get MFCCs per frame')
    mfccs = [np.mean(librosa.feature.mfcc(y=frame).T,axis=0) for frame in tqdm(audio_by_frame)]

    print('=> MFCC Length: {}'.format(len(mfccs)))
    print('=> Length of one MFCC: {}'.format(len(mfccs[0])))
    np.savetxt(Path(video_path.parent, '{}.audio.gz'.format(video_path.stem)), mfccs)

# compile into one massive dataset?

# make sure data loaded properly
# data_path = 'audioFrames.gz'
# audio_data = np.loadtxt(data_path)
# print(audio_data[0])

# test = np.load('data_2d_custom_myvideos.npz', allow_pickle=True)
# test['positions_2d'], nah we want 3d
# test = np.load('keypoints.npz.npy', allow_pickle=True) gives array of the keypoints, what we need

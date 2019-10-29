from __future__ import unicode_literals
from pathlib import Path
import youtube_dl
import numpy as np
import librosa
import subprocess
from imutils.video import count_frames
from tqdm import tqdm

# make command line arguments for playlist url, separate av, and extract audio
playlist_url = 'https://www.youtube.com/watch?v=htLt1UvEMLE&list=PL8Kqpe3GkRgGNlb-RYgT8oBmPLuajZu4X'
separate_AV = False
extract_audio = True

# prepare path
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

if extract_audio:
  videos = Path(download_dir).rglob('*.mp4')
  for video_path in videos:
    print('\n=> {}'.format(video_path))
    print("=> Counting Video Frames")
    num_frames = count_frames(str(video_path))
    print('=> Video Frames: {}'.format(num_frames))

    print('=> Extracting Audio')
    audio_path = Path(video_path.parent, '{}.wav'.format(video_path.stem))
    command = "ffmpeg -hide_banner -loglevel panic -y -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(video_path, audio_path)
    subprocess.call(command, shell=True)

    print('=> Loading Audio into librosa')
    audio_data, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')

    print("=> Group audio data by frame")
    audio_by_frame = np.array_split(audio_data, num_frames)

    print("=> Get MFCCs per frame")
    mfccs = [np.mean(librosa.feature.mfcc(y=frame).T,axis=0) for frame in tqdm(audio_by_frame)]

    print("=> MFCC Length: {}".format(len(mfccs)))
    print("=> Length of one MFCC: {}".format(len(mfccs[0])))
    np.savetxt(Path(video_path.parent, '{}.gz'.format(video_path.stem)), mfccs)

# make sure data loaded properly
# data_path = 'audioFrames.gz'
# audio_data = np.loadtxt(data_path)
# print(audio_data[0])
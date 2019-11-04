#from __future__ import unicode_literals
from pathlib import Path
import youtube_dl
import argparse

def main(args):
  playlist_url = args.playlist_url if args.playlist_url != None else 'https://www.youtube.com/watch?v=htLt1UvEMLE&list=PL8Kqpe3GkRgGNlb-RYgT8oBmPLuajZu4X'
  separate_AV = args.separate_AV
  extract_audio = not separate_AV

  # prepare path for data
  label = args.label
  download_dir = Path(Path.cwd(), 'data/', label)
  download_dir.mkdir(parents=True, exist_ok=True) 

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
      for file_path in folder.glob('*'):
        file_path.rename(folder, str(count).zfill(5) + file_path.suffix)
      folder.rename(Path(folder.parent, str(count).zfill(5)))
      count += 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Downloads videos to be used in dataset')
  parser.add_argument('label', type=str,
                      help='Label for the dataset (e.x. Popping)')
  parser.add_argument('--playlist_url', type=str,
                      help='The YouTube playlist to get videos from (default is the Kakashi Raw Playlist)')
  parser.add_argument('--separate_AV', action='store_true',
                      help='Prevent muxing of Audio/Video into one stream (used when videos don\'t need to be cut)')
  args = parser.parse_args()
  main(args)
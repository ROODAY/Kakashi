from __future__ import unicode_literals
from pathlib import Path
import youtube_dl
import os
import sys, getopt

def main(argv):
  video_url = ''

  try:
    opts, args = getopt.getopt(argv,"h",["url="])
  except getopt.GetoptError:
    print ('download.py --url=<video url>')
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':
      print ('download.py --url=<video url>')
      sys.exit()
    elif opt == "--url":
      video_url = arg

  downloadDir = os.path.join(os.getcwd(), 'individual')
  Path(downloadDir).mkdir(exist_ok=True) 

  ydl_opts = {
    'outtmpl': 'individual/%(title)s.%(ext)s',
    'restrictfilenames': True,
    'writeinfojson': False,
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]' 
  }

  with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

if __name__ == "__main__":
  main(sys.argv[1:])
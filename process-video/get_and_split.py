from __future__ import unicode_literals
from pathlib import Path
from moviepy.editor import *
import youtube_dl
import os
import cv2

downloadDir = os.path.join(os.getcwd(), 'videos')
Path(downloadDir).mkdir(exist_ok=True) 

ydl_opts = {
	'outtmpl': 'videos/%(id)s/%(id)s.%(ext)s',
	'restrictfilenames': True,
    'writeinfojson': False,
    'format': 'bestvideo[ext=mp4],bestaudio[ext=m4a]',
}

# download playlist
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=VKrBSVUL1so&list=PLVn0x_fJ3ZS-uPTjf4flLRUhR32xd0RBB'])

# split videos into frames and audio
# videos = Path(downloadDir).glob('**/*.mp4')
# for video_path in videos:
# 	video_dir = os.path.dirname(video_path)
# 	frame_dir = os.path.join(video_dir, 'frames')
# 	Path(frame_dir).mkdir(exist_ok=True)

# 	audioclip = AudioFileClip(str(video_path))
# 	audioclip.write_audiofile(str(os.path.join(video_dir, 'audio.wav')))

# 	vidcap = cv2.VideoCapture(str(video_path))
# 	success, image = vidcap.read()
# 	count = 0
# 	while success:
# 		frame_path = os.path.join(frame_dir, '%d.jpg' % count)
# 		# print("writing to: ", frame_path)
# 		cv2.imwrite(frame_path, image)
# 		success, image = vidcap.read()
# 		count += 1
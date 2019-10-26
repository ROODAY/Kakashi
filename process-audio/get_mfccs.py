import numpy as np
import librosa
import os
import subprocess
from imutils.video import count_frames
from tqdm import tqdm

print("=> Extract Audio")
video_path = 'cut.mp4'
audio_path = 'audio.wav'
result_path = 'audioFrames.gz'
command = "ffmpeg -i cut.mp4 -ab 160k -ac 2 -ar 44100 -vn {}".format(audio_path)
subprocess.call(command, shell=True)

print('=> Load Audio')
audio_data, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')

print("=> Count Video Frames")
num_frames = count_frames(video_path)
print('=> Video Frames: {}'.format(num_frames))

print("=> Group audio data by frame")
audio_by_frame = np.array_split(audio_data, num_frames)

print("=> Get MFCCs per frame")
mfccs = [np.mean(librosa.feature.mfcc(y=frame).T,axis=0) for frame in tqdm(audio_by_frame)]

print("=> MFCC Length: {}".format(len(mfccs)))
print("=> Length of one MFCC: {}".format(len(mfccs[0])))
np.savetxt(result_path, mfccs)
os.remove(audio_path)
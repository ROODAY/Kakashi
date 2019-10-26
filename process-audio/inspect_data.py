import numpy as np

data_path = 'audioFrames.gz'
audio_data = np.loadtxt(data_path)
print(audio_data[0])
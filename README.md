# Project Kakashi

Directed Study project, Fall 2019

Goal: Create a model that generate dance choreography from arbitrary music

Current plan:
- Find/Create playlist on YouTube or some set of suitable videos
- Download playlist/set. For each video in set, extract audio and frames (frames may need to be normalized/processed further for the model, also to remove unneccessary frames)
- For each video, go through frames and estimate pose (save as 3-D coordinates, normalized to the same origin)
- Now with the training data, try naive solution of recurrent neural network. Each step takes as input: current frame audio features, previous frame audio features, previous frame pose matrix. It will then output a pose matrix for the current frame. The current frame's audio and pose are fed into the next step, etc.
- After training, it should be able to take a sequence of frames and output a series of pose matrices, which can then be animated.

Segments:
- [ ] Create training data (download/process/etc.)
- [ ] Create RNN to guess pose for current audio feature
- [ ] Improve model to use a GAN (look at C-RNN-GAN etc.)
- [ ] Animate resultant pose matrices for choreography


Stuff to tie together to get training data:
- Run download.py in process-video to get a bunch of videos
- Cut them to make sure only dancing is in/1 person in the video
- get MFCC data for all of them 
- get poses 
- make sure len of poses == len of mfcc per video

Possible Changes:
- Use something other than mfcc as representation of the audio
- perhaps get multi body working with videpose

Make sure the following environment variables for directory roots are set:
- KAKASHI
- VIDEOPOSE
- DETECTRON


need train, validation, and test set


figure out why data is converging

get good dataset
make sure input is differentiable - done


try removing dropout - slight improvement, tried putting back in with dropout2d for decoder

possibly due to eos and sos - didn't do much, made loss a bit worse

try removing padding since batch size of 1 - didn't do much, but keep this as it  makes sense
get rid of gradient clip - didn't do much, put back in
reverse src - doesn't do much, removed as it doesnt make sense here

mseloss borke? - no

try other loss function - huber
another optimizer - sgd
try batching - need more data first
keep pose in flattenend form - don't think this will do anything

get song length before feature extraction, then stop inference when length is reached (length * 30)
perhaps just need more data, look into flipping

python3 animate.py -d custom -k kakashi -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject 00001.mp4 --viz-action custom --viz-camera 0 --viz-video /project/dnn-motion/kakashi/Kakashi/data/test/00001/00001.mp4 --viz-output 00001.test.mp4 --viz-size 6

get freestyle practice videos, ask dancers your know if they can send videos preferably in mp4 format/in bulk. perhaps a thing for next semester, getting funding? if I could set parameters on the dataset it could be a lot better

seq2seq long - whole song to whole dance
seq2seq short - subsection of song to short section of dance (configurable time interval like 5s 10s 15s etc)
per frame - break up song into frames (30 fps) and do frame by frame training

for inference, the first pose should be start of sequence. we can do all 0s or 1 or something, just to kick it off
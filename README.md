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
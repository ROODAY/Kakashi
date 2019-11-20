# Project Kakashi
Generate dance choreography (in the form of 3D pose matrices) from arbitrary music samples.

Current setup:
- Download playlists of dance videos from YouTube
- Cut videos to remove unnecessary information (intros/outros/etc.)
- Run 3D pose estimation to get "ground truth" dataset output
- Extract audio features from videos to serve as dataset input
- Train a sequence to sequence model to generate pose from music features

Tasks
- [x] Create training data (download/process/etc.)
- [x] Create RNN to guess pose for current audio feature
- [ ] Improve model architecture (4 layer LSTM, 1024 hidden dim)
- [ ] Try other RNN architectures
- [ ] Create config files for different architecture params
- [ ] Try other loss functions/optimizers
- [ ] Try other input audio features
- [ ] Animate results (create standalone file to drop into VideoPose)


## Installation
Make sure the following environment variables for directory roots are set:
- KAKASHI
- VIDEOPOSE
- DETECTRON

## Training/Inference
Animating results:
```
python3 animate.py -d custom -k kakashi -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject 00001.mp4 --viz-action custom --viz-camera 0 --viz-video /project/dnn-motion/kakashi/Kakashi/data/test/00001/00001.mp4 --viz-output 00001.test.mp4 --viz-size 6
```

## Experiments
- [ ] Generate entire song length choreography at a time
- [ ] Generate a small interval of pose at a time until song is complete (5s, 10s, etc.) (last frame of output is seed for next interval)
- [ ] Generate a frame of pose at a time until song is complete (output frame is seed for next frame)
- [ ] Infer by generating frame by frame (total frames is 30 * song_length)
- [ ] Flip pose data horizontally to double dataset (check how VideoPose renders)
- [ ] Other datasets (freestyle hip-hop, kpop, etc.)
- [ ] Find a T-pose to use as seed pose for inference
- [ ] Use a random first frame pose as seed pose for inference
- [ ] For intervals/frame, one video is no longer a data point but a batch

## Future Work
- Output multi-body pose
- Other model archictecures (not sequence to sequence)
- High quality dataset (record dancers)
ormat/in bulk. perhaps a thing for next semester, getting funding? if I could set parameters on the dataset it could be a lot better

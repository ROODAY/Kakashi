#!/bin/bash -l

# Set SCC project
#$ -P dnn-motion

# Request 4 CPUs
#$ -pe omp 1

# Request 1 GPU (the number of GPUs needed should be divided by the number of CPUs requested above)
#$ -l gpus=4

# Specify the minimum GPU compute capability 
#$ -l gpu_c=3.5

#$ -N test-infer-video
#$ -j y 

#$ -M rooday@bu.edu

module load ffmpeg/4.0.3
module load python3/3.7.3
module load cuda/10.1
module load pytorch/1.1
module load geos/3.7.0

cd /project/dnn-motion/kakashi/detectron/
python3 tools/infer_video.py --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir ../video_output --image-ext mp4 --wts models/model_final.pkl ../video_input

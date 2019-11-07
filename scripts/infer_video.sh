#!/bin/bash -l

#$ -P dnn-motion
#$ -l gpus=4
#$ -l gpu_c=3.5
#$ -N kakashi-2d-pose
#$ -j y 
#$ -M rooday@bu.edu

module load ffmpeg/4.2.1
module load python3/3.7.3
module load cuda/10.1
module load pytorch/1.1
module load geos/3.7.0

cd /project/dnn-motion/kakashi/detectron/
python3 tools/infer_video.py --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir /project/dnn-motion/kakashi/Kakashi/tmp_out --image-ext mp4 --wts models/model_final.pkl /project/dnn-motion/kakashi/Kakashi/tmp_in

#!/bin/bash -l

#$ -P dnn-motion
#$ -l gpus=4
#$ -l gpu_c=3.5
#$ -N kakashi-generate-dataset
#$ -j y 
#$ -M rooday@bu.edu

module load ffmpeg/4.0.3
module load python3/3.7.3
module load cuda/10.1
module load pytorch/1.1
module load geos/3.7.0

cd /project/dnn-motion/kakashi/Kakashi
python3 tools/generate_dataset.py test

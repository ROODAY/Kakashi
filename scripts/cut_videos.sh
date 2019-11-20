#!/bin/bash -l

#$ -P dnn-motion
#$ -l gpus=1
#$ -l gpu_c=3.5
#$ -N kakashi-cut-videos
#$ -m ae
#$ -M rooday@bu.edu
#$ -j y
#$ -o cut-videos.log
#$ -V

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load ffmpeg/4.2.1
module load python3/3.7.3
module load cuda/10.1

cd $KAKASHI
python3 tools/cut_videos.py wod

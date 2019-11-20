#!/bin/bash -l

#$ -P dnn-motion
#$ -N kakashi-download-videos
#$ -m ae
#$ -M rooday@bu.edu
#$ -j y
#$ -o download-videos.log
#$ -V

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load ffmpeg/4.2.1
module load python3/3.7.3

cd $KAKASHI
python3 tools/download_playlist.py wod

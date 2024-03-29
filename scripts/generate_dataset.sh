#!/bin/bash -l

#$ -P dnn-motion
#$ -l h_rt=160:00:00
#$ -l gpus=4
#$ -l gpu_c=3.5
#$ -N kakashi-generate-dataset
#$ -m ae
#$ -M rooday@bu.edu
#$ -j y
#$ -o generate-dataset.log
#$ -V

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load ffmpeg/4.2.1
module load python3/3.7.3
module load cuda/10.1
module load pytorch/1.1
module load geos/3.7.0

cd $KAKASHI
python3 tools/generate_dataset.py test --videpose_dataset_name kakashi-test
python3 tools/generate_dataset.py test --skip_detect_pose --skip_extract --audio_feature mfcc-frame

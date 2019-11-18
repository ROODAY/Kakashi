#!/bin/bash -l

#$ -P dnn-motion
#$ -l gpus=4
#$ -l gpu_c=3.5
#$ -N kakashi-generate-dataset
#$ -m ae
#$ -M rooday@bu.edu
#$ -j y
#$ -o generate-dataset-output.txt
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

cd /project/dnn-motion/kakashi/Kakashi
python3 tools/generate_dataset.py test-cut

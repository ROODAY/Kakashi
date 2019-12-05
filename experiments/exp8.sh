#!/bin/bash -l

#$ -P dnn-motion
#$ -l gpus=1
#$ -l gpu_c=3.5
#$ -N kakashi-experiment-8
#$ -m ae
#$ -M rooday@bu.edu
#$ -j y
#$ -o experiment-8.log
#$ -V

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load python3/3.7.3
module load cuda/10.1
module load pytorch/1.1
module load geos/3.7.0

cd $KAKASHI
python3 train.py wod --deterministic --hide_tqdm --model_name exp8 --config config/exp8.yaml --load_iterators its/checkpoint.pkl

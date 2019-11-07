#!/bin/bash -l

#$ -P dnn-motion
#$ -l gpus=4
#$ -l gpu_c=3.5
#$ -N kakashi-2d-pose
#$ -m ae
#$ -M rooday@bu.edu
#$ -j y
#$ -o 2d-pose-output.txt
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

cd /project/dnn-motion/kakashi/detectron/
python3 tools/infer_video.py --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir /project/dnn-motion/kakashi/Kakashi/tmp_out --image-ext mp4 --wts models/model_final.pkl /project/dnn-motion/kakashi/Kakashi/tmp_in

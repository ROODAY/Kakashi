#!/bin/bash -l

#$ -P dnn-motion
#$ -l gpus=1
#$ -l gpu_c=3.5
#$ -N kakashi-render-experiments
#$ -m ae
#$ -M rooday@bu.edu
#$ -j y
#$ -o render-experiments.log
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
python3 infer.py pre/exp1.best_valid.pt data/leopop.wav --config config/exp1.yaml --render --render_name render/exp1.best_valid.mp4
python3 infer.py pre/exp1.trained.pt data/leopop.wav --config config/exp1.yaml --render --render_name render/exp1.trained.mp4

python3 infer.py pre/exp2.best_valid.pt data/leopop.wav --config config/exp2.yaml --render --render_name render/exp2.best_valid.mp4
python3 infer.py pre/exp2.trained.pt data/leopop.wav --config config/exp2.yaml --render --render_name render/exp2.trained.mp4

python3 infer.py pre/exp3.best_valid.pt data/leopop.wav --config config/exp3.yaml --render --render_name render/exp3.best_valid.mp4
python3 infer.py pre/exp3.trained.pt data/leopop.wav --config config/exp3.yaml --render --render_name render/exp3.trained.mp4

python3 infer.py pre/exp4.best_valid.pt data/leopop.wav --config config/exp4.yaml --render --render_name render/exp4.best_valid.mp4
python3 infer.py pre/exp4.trained.pt data/leopop.wav --config config/exp4.yaml --render --render_name render/exp4.trained.mp4

python3 infer.py pre/exp5.best_valid.pt data/leopop.wav --config config/exp5.yaml --render --render_name render/exp5.best_valid.mp4
python3 infer.py pre/exp5.trained.pt data/leopop.wav --config config/exp5.yaml --render --render_name render/exp5.trained.mp4

python3 infer.py pre/exp6.best_valid.pt data/leopop.wav --config config/exp6.yaml --render --render_name render/exp6.best_valid.mp4

python3 infer.py pre/exp8.best_valid.pt data/leopop.wav --config config/exp8.yaml --render --render_name render/exp8.best_valid.mp4
python3 infer.py pre/exp8.trained.pt data/leopop.wav --config config/exp8.yaml --render --render_name render/exp8.trained.mp4

python3 infer.py pre/exp9.best_valid.pt data/leopop.wav --config config/exp9.yaml --render --render_name render/exp9.best_valid.mp4
python3 infer.py pre/exp9.trained.pt data/leopop.wav --config config/exp9.yaml --render --render_name render/exp9.trained.mp4

python3 infer.py pre/exp10.best_valid.pt data/leopop.wav --config config/exp10.yaml --render --render_name render/exp10.best_valid.mp4
python3 infer.py pre/exp10.trained.pt data/leopop.wav --config config/exp10.yaml --render --render_name render/exp10.trained.mp4

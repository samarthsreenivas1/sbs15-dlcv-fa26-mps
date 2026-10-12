#!/bin/bash

### Do not change the following 7 lines (job name / account / partition / resources).
#SBATCH --job-name=flower_demo
#SBATCH --account=bfdu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=48g
### End Do not change block

# Wall-clock limit. The demo run finishes in about 6 minutes on a GH200. Raise this
# only if your own run really needs it: short requests start much sooner,
# while 2-hour requests get held (QOSGrpBillingMinutes) once the class
# allocation balance is low.
#SBATCH --time=00:45:00

#SBATCH --mail-user=NETID@illinois.edu  # replace with your email
#SBATCH --mail-type=NONE                # Type of email notifications to send

#SBATCH --output=ece494mp3.1-%x-%j.out  # Default name of batch job output file if not set on command line
#SBATCH --error=ece494mp3.1-%x-%j.err   # Default name of batch job error file if not set on command line

# Make the job fail if python fails: without pipefail `| tee` hides python's
# exit code and slurm reports a crashed run as COMPLETED. Keep the python
# command below as the last command of this script so its status becomes the
# job's exit status.
set -o pipefail

# OUTPUT_DIR and EXP_NAME are passed in from the command line via
#   sbatch --export=ALL,OUTPUT_DIR=...,EXP_NAME=... (see README.md).
# Use a different OUTPUT_DIR for every run, or the new run overwrites the
# previous one. EXP_NAME selects the hyperparameter block in config.yaml, so
# every new EXP_NAME needs its own block there.
OUTPUT_DIR=${OUTPUT_DIR:="runs_flower/vit_linear"}
EXP_NAME=${EXP_NAME:="vit_linear"}

mkdir -p ${OUTPUT_DIR}
LOG_FILE="${OUTPUT_DIR}/log.txt"

module load python/3.10.14

# Activate your venv
source /work/nvme/bfdu/cs444/fa26/mp3/venv/bin/activate

# print some diagnostics
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
export PYTHONUNBUFFERED=1

# Run your job 

python demo_flower.py --exp_name $EXP_NAME \
    --output_dir ${OUTPUT_DIR} \
    --data_dir /work/nvme/bfdu/cs444/fa26/mp3/data/flower-dataset-reduced/ 2>&1 | tee ${LOG_FILE}

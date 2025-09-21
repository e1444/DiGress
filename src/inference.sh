#!/bin/bash
#SBATCH --account=def-six
#SBATCH --job-name=DiGress_Inference
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=10:00:00
#SBATCH --output=slurm-logs/%j/log.out
#SBATCH --error=slurm-logs/%j/log.err

module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load python/3.11.5
module load rdkit/2023.09.3
module load cuda/12.2

export LD_LIBRARY_PATH=/home/e1444/.local/lib:$LD_LIBRARY_PATH

virtualenv --no-download --system-site-packages ENV --python=python3.11
source ENV/bin/activate

pip install -r requirements.txt
pip install -e .

CHECKPOINT_DIR="/home/e1444/repos/DiGress/checkpoints/moses"
CHECKPOINT_FILE="$CHECKPOINT_DIR/moses_last-v1.ckpt"

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Checkpoint file not found. Downloading..."
    mkdir -p "$CHECKPOINT_DIR"
    pip install gdown
    gdown --id 1LUVzdZQRwyZWWHJFKLsovG9jqkehcHYq -O "$CHECKPOINT_FILE"
    if [ $? -ne 0 ]; then
        echo "Failed to download the checkpoint file."
        exit 1
    fi
    echo "Checkpoint downloaded successfully to $CHECKPOINT_FILE"
else
    echo "Checkpoint file already exists at $CHECKPOINT_FILE"
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

srun --jobid=${SLURM_JOB_ID} --ntasks=1 nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 > slurm-logs/${SLURM_JOB_ID}/gpu_usage.csv &
MONITOR_PID=$!

BATCHES=20
BATCH_SIZE=512

time python src/main.py \
    +experiment=moses.yaml \
    dataset=moses.yaml \
    general.gpus=1 \
    general.test_only=$CHECKPOINT_FILE \
    general.resume=$CHECKPOINT_FILE \
    general.final_model_samples_to_generate=$(($BATCHES * $BATCH_SIZE)) \
    train.batch_size=$BATCH_SIZE \
    train.num_workers=4

kill $MONITOR_PID
#!/bin/bash
#SBATCH --account=def-six
#SBATCH --job-name=DiGress_Inference
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=00:05:00

module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load rdkit/2023.09.3
module load cuda/12.6

export LD_LIBRARY_PATH=/home/e1444/.local/lib:$LD_LIBRARY_PATH

virtualenv --no-download --system-site-packages ENV --python=python3.11
source ENV/bin/activate

pip install -r requirements.txt
pip install -e .

python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi
python -c "import graph_tool; print('graph-tool version:', graph_tool.__version__)"

echo "---"
echo "Running a simple CUDA test..."
python -c "
import torch
if torch.cuda.is_available():
    print('CUDA is available. Let\\'s test it.')
    try:
        # Create a tensor and move it to the GPU
        x = torch.tensor([1.0, 2.0]).cuda()
        # Do a simple operation
        y = x * 2
        # Move it back to CPU to print
        print('CUDA test successful. Result on GPU:', y)
        print('Result on CPU:', y.cpu())
    except Exception as e:
        print('CUDA test FAILED with error:')
        print(e)
else:
    print('CUDA is not available.')
"
echo "---"

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export BLIS_NUM_THREADS=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# python src/main.py \
#     +experiment=moses.yaml \
#     dataset=moses.yaml \
#     general.gpus=1 \
#     train.batch_size=512 \
#     train.num_workers=4 \
#     general.test_only=/home/e1444/repos/DiGress/checkpoints/moses/moses_last-v1.ckpt
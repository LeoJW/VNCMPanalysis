#!/bin/bash
#SBATCH -Jrun_main_redundancy                   # Job name
#SBATCH --account=gts-ssponberg3-paid           # charge account
#SBATCH -N1                                     # Number of nodes and cores per node required
#SBATCH -G1 -C'L40S|A100|V100-32GB'             # Accept L40S, A100 80GB, or V100 32GB GPUs
#SBATCH --mem-per-gpu=32G                       # Memory per GPU
#SBATCH -t4300                                  # Duration of the job in minutes
#SBATCH -qinferno                               # QOS Name
#SBATCH --array=0-11                            # Array range for 12 tasks
#SBATCH --output=main_redundancy_report.%A\_%a.log
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=leo.w@gatech.edu            # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory
module purge
module load cuda/12.1.1
module load pytorch/2.1.0                       # Load module dependencies
srun --unbuffered apptainer exec --nv $PYTORCH_2_1_0_SIF python run_main_redundancy.py $SLURM_ARRAY_TASK_ID
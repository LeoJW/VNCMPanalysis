#!/bin/bash
#SBATCH -Jrun_main_single_neurons               # Job name
#SBATCH --account=gts-ssponberg3-paid           # charge account
#SBATCH -N1                                     # Number of nodes and cores per node required
#SBATCH --gres=gpu:1                            # GPU type (any) and number of GPUs
#SBATCH --mem-per-gpu=16G                       # Memory per GPU
#SBATCH -t3000                                  # Duration of the job in minutes
#SBATCH -qinferno                               # QOS Name
#SBATCH --output=main_single_neurons_report.%A\_%a.log
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=leo.w@gatech.edu            # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory
module purge
module load cuda/12.1.1
module load pytorch/2.1.0                       # Load module dependencies
srun --unbuffered apptainer exec --nv $PYTORCH_2_1_0_SIF python run_main_quicktest.py $SLURM_ARRAY_TASK_ID
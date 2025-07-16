#!/bin/bash
#SBATCH -Jrun_precision_kinematics              # Job name
#SBATCH --account=gts-ssponberg3                # charge account
#SBATCH -N1                                     # Number of nodes and cores per node required
#SBATCH --gres=gpu:L40S:1                       # GPU type (any) and number of GPUs
#SBATCH --mem-per-gpu=12G                       # Memory per GPU
#SBATCH -t3000                                  # Duration of the job in minutes
#SBATCH -qinferno                               # QOS Name
#SBATCH --output=kinematics_precision_report.%A\_%a.log
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=leo.w@gatech.edu            # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory
module purge
module load cuda/12.1.1
module load pytorch/2.1.0                       # Load module dependencies
srun apptainer exec --nv $PYTORCH_2_1_0_SIF python run_precision_kinematics.py $SLURM_ARRAY_TASK_ID
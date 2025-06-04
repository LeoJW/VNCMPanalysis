#!/bin/bash
#SBATCH -run_network_comparison                 # Job name
#SBATCH --account=gts-ssponberg3                # charge account
#SBATCH -N1                                     # Number of nodes and cores per node required
#SBATCH --gres=gpu:V100:1                       # GPU type (V100) and number of GPUs
#SBATCH --mem-per-gpu=20G                       # Memory per GPU
#SBATCH -t900                                   # Duration of the job in minutes
#SBATCH -qinferno                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=leo.w@gatech.edu            # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load pytorch/2.1.0                       # Load module dependencies
srun python run_network_comparison.py
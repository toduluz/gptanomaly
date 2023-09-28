#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=10           # Number of CPU to request for the job
#SBATCH --mem=32GB                   # How much memory does your job require?
#SBATCH --gres=gpu:4                # Do you require GPUS? If not delete this line
#SBATCH --time=05-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=/common/home/users/y/yingfu.lim.2022/anomalylog/%u.%j.out         # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=dgxv100                 # The partition you've been assigned
#SBATCH --account=guansongresearch   # The account you've been assigned (normally student)
#SBATCH --qos=normal       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=yingfu.lim.2022@msc.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=test_bgl_v0     # Give the job a name
#SBATCH --requeue

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Python/3.11.4 

# Create a virtual environment
# python3 -m venv ~/myenv
# python3 -m venv ~/myenv

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source ~/anomalylog/myenv/bin/activate

# Find out which GPU you are using
srun whichgpu

# If you require any packages, install it as usual before the srun job submission.
# pip3 install -r requirements.txt
# pip3 install -r requirements.txt

# Submit your job to the cluster
srun --gres=gpu:4 bash running_scripts/fine_tuning/train_logs_ad.sh
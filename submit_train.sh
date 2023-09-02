#!/bin/bash
#SBATCH --job-name=train-cdcr-gpu
#SBATCH -t 24:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared              # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                   # take 1 GPU, see https://www.hlrn.de/doc/display/PUB/GPU+Usage for more options
#SBATCH --mem-per-gpu=30G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=4            # number cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=daniel.tranortega01@stud.uni-goettingen.de  # TODO: change this to your mailaddress!
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

source ~/.bashrc


module load cuda

which python 
echo $PATH

conda info --envs
source activate dnlp # 


# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"


export http_proxy=http://www-cache.gwdg.de:3128
export https_proxy=http://www-cache.gwdg.de:3128

echo "added http proxy"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script:
#Pretrain setting
#python -u classifier.py --option pretrain --use_gpu
#finetune setting
#python -u classifier.py --option finetune --use_gpu --epochs 30 --batch_size 128 --lr 1e-5
#python -u multitask_classifier.py --option pretrain --lr 1e-3 --batch_size 64 --local_files_only --use_gpu
python -u multitask_classifier.py --option finetune --lr 1e-5 --batch_size 64 --local_files_only --use_gpu --epochs 50
# Run the script with logger:
#python -u train_with_logger.py -l ~/${SLURM_JOB_NAME}_${SLURM_JOB_ID}  -t True -p True -d True -s True -f True

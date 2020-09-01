#!/bin/bash 

#PBS -q cccr
#PBS -N conv1
#PBS -l select=1:ncpus=36:vntype=cray_compute
#PBS -l walltime=48:00:00
#PBS -l place=scatter

cd $PBS_O_WORKDIR

module load anaconda/3.7
module load craype-broadwell
source activate knp_ai

export OMP_NUM_THREADS=72
aprun -n 1 -j 2 -d $OMP_NUM_THREADS /home/cccr/supriyo/.conda/envs/knp_ai/bin/python conv1d_RMM2_parrallel.py   >> output_rmm2.log

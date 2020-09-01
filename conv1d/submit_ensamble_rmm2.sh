#!/bin/bash 

#PBS -q cccr
#PBS -N ensamble_rmm2
#PBS -l select=1:ncpus=36:vntype=cray_compute
#PBS -l walltime=48:00:00
#PBS -l place=scatter

cd $PBS_O_WORKDIR

module load craype-broadwell
source activate knp_ai

export OMP_NUM_THREADS=72
aprun -n 1 -j 2 -d $OMP_NUM_THREADS /home/cccr/supriyo/.conda/envs/knp_ai/bin/python ensamble_rmm2.py   >> output_ensamble2_rmm2.log

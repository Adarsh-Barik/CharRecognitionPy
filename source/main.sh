#!/bin/bash -l
# FILENAME: main.sh
module load devel
module load anaconda/2.0.1-py27
ipython
cd $PBS_O_WORKDIR

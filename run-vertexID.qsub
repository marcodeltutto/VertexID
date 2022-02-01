#!/bin/bash -l
#COBALT -t 150
#COBALT -n 3
#COBALT -A datascience
#COBALT -q single-gpu

module load conda/2021-11-30
conda activate
module load hdf5

source /lus/grand/projects/datascience/cadams/thetagpu-conda-2021-11-20/bin/activate

WORKDIR=/home/cadams/ThetaGPU/VertexID

cd $WORKDIR

N_NODES=$(cat $COBALT_NODEFILE | wc -l)
RANKS_PER_NODE=8
let N_RANKS=${RANKS_PER_NODE}*${N_NODES}

let MB=${N_RANKS}*128


mpirun -n $N_RANKS -hostfile ${COBALT_NODEFILE} -map-by node \
-x VIRTUAL_ENV \
-x PATH \
-x LD_LIBRARY_PATH \
-x PYTHONSTARTUP \
-x PYTHONUSERBASE \
python bin/exec.py \
run.distributed=True \
run.minibatch_size=${MB} \
data.image_mode=sparse \
run.iterations=5000 \
data.aux_file=non \
data.downsample=0 \
run.id=sparse_yolo-real

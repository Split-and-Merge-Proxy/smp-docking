set -x

PARTITION=$1
NODES=$2
JOB_NAME=$3
GPUS=$4
GPUS_PER_NODE=$5
CPUS_PER_TASK=$6
QUOTATYPE=${QUOTATYPE:-'reserved'}

srun -p ${PARTITION} \
    --nodes=${NODES} \
    --job-name=${JOB_NAME} \
    --quotatype=${QUOTATYPE} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -m src.inference_rigid
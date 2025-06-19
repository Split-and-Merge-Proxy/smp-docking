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
    python -m src.train -graph_residue_loc_is_alphaC -cross_msgs -use_dist_in_layers -use_edge_features_in_gmn -use_mean_node_features \
                        -lr 3e-4 -data 'dips_het' -data_fraction 0.2 -patience 30 -warmup 1.0 -split 0 -method smp -resume_ckpt /mnt/petrelfs/duhao.d/projects/smp-docking/equidock_public/checkpts/smp_pretrain/pseudo_multi_model_best.pth
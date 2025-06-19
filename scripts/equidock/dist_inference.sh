set -x


CUDA_VISIBLE_DEVICES=0 python -m src.inference_rigid --method_name equidock --dataset dips_het \
                                --ckpt_path ./checkpts/equidock_dips_het/dips_het_model_best.pth
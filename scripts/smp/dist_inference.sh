set -x

CUDA_VISIBLE_DEVICES=0  python -m src.inference_rigid --method_name smp --dataset dips_het --ckpt_path ./checkpts/smp/dips_het_model_best.pth
set -x

echo 'Start Inference...'
CUDA_VISIBLE_DEVICES=0  python -m src.inference_rigid --method_name smp --dataset dips_het --ckpt_path /your/own/local/path/to/test ckpt

sleep 5

echo 'Start Evaluation...'
python -m src.test_all_methods.eval_pdb_outputset --dataset dips_het --method_name smp --data_frac 1.0
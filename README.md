# SMP-docking 
This repository is for the rigid protein-protein docking task.

## 1. Environment Setup

```bash
git clone https://github.com/Split-and-Merge-Proxy/smp-docking.git
cd smp-docking
conda create -n smp-docking python=3.9
conda activate smp-docking
pip install -r requirements.txt
```

## 2. Data Preparation
You can download the docking data from [SMP - Harvard Dataverse](https://doi.org/10.7910/DVN/22AUPR) and place it in the `./cache` folder.

## 3. Training (Optional)
### EquiDock
```bash
# Pytorch DDP
bash ./scripts/equidock/dist_train.sh
# Slurm 
bash ./scripts/equidock/slurm_train.sh
```
**Note:** you can change the `data_fraction` in the Shell file to determine the amount of training data.

### SMP
```bash
# Pytorch launcher
bash ./scripts/smp/dist_pretrain.sh
bash ./scripts/smp/dist_finetune.sh

# Slurm launcher
bash ./scripts/smp/slurm_pretrain.sh
bash ./script/smp/slurm_finetune.sh
```
**Note:** you can change the `data_fraction` in the Shell file to determine the amount of fine-tuning data (pre-training data does not support yet) and `resume_ckpt` to your own directory.



## 4. Evaluations
### EquiDock
```bash
# Pytorch launcher
bash ./scripts/equidock/dist_inference.sh
bash ./scripts/equidock/dist_eval.sh
# Slurm launcher
bash ./scripts/equidock/slurm_inference.sh
bash ./scripts/equidock/slurm_eval.sh
```

### SMP
```bash
# Pytorch launcher
bash ./scripts/smp/dist_inference.sh
bash ./scripts/smp/dist_eval.sh
# Slurm launcher
bash ./scripts/smp/slurm_inference.sh
bash ./scripts/smp/slurm_eval.sh
```

## Acknowledges
- [EquiDock](https://github.com/octavian-ganea/equidock_public)
- [EBMDock](https://github.com/wuhuaijin/EBMDock)
- [HMR](https://github.com/bytedance/HMR)
- [DIPS](https://github.com/drorlab/DIPS)


If you have any questions, please don't hesitate to contact me through [cs.dh97@gmail.com](cs.dh97@gmail.com)
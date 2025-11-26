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
You can download the docking data (`dips_het_residues_maxneighbor_10_cutoff_30.0_pocketCut_8.0.zip` for DIPS-Het dataset and `pseudo_dimer_residues_maxneighbor_10_cutoff_30.0_pocketCut_8.0.zip` for pseudo-dimer pre-training dataset) from [SMP - Harvard Dataverse](https://doi.org/10.7910/DVN/0QURCP), and place them in the `./cache` directory.

## 3. Training (Optional)
### EquiDock
```bash
bash ./scripts/equidock_train.sh
```
**Note:** you can change the `data_fraction` in the Shell file to determine the amount of training data.

### SMP
```bash
bash ./scripts/smp_pretrain.sh
bash ./scripts/smp_finetune.sh
```
**Note:** you can change the `data_fraction` in the Shell file to determine the amount of fine-tuning data (pre-training data does not support yet) and the `resume_ckpt` in the Shell file to select the pre-trained ckpt to your own path.



## 4. Evaluations

```bash
bash ./scripts/test.sh
```
**Note:** You can change the `method_name` in the `test.sh` to determine whether eval the EquiDock or SMP method, and change the `ckpt_path` in the `test.sh` to its corresponding ckpt path.

## 5. Reproducing the Results Reported in the Manuscript
To reproduce the results reported in our manuscript, first download the processed test set (`dips_het_residues_maxneighbor_10_cutoff_30.0_pocketCut_8.0.zip`) from https://doi.org/10.7910/DVN/0QURCP and unzip it.

Then, change the `method_name` in the test.sh script to point to your local path for processed test sets and `ckpt_path` for the ckpt path, and run the following command:

```bash
bash ./scripts/test.sh
```

The expected results are shown below.

**DIPS-Het**
||   | Complex RMSD |  |  | Interface RMSD |  |  |
|----------|----------|----------|----------|----------|----------|----------|----------|
|Method| Median  | Mean | Std| Median | Mean | Std | Success Rate |
|----------|----------|----------|----------|----------|----------|----------|----------|
|EquiDock|  |  |   |  |  |  |  |
|SMP|   |   |   |  |  |  | |

## Acknowledges
- [EquiDock](https://github.com/octavian-ganea/equidock_public)
- [EBMDock](https://github.com/wuhuaijin/EBMDock)
- [HMR](https://github.com/bytedance/HMR)
- [DIPS](https://github.com/drorlab/DIPS)


If you have any questions, please don't hesitate to contact me through [cs.dh97@gmail.com](cs.dh97@gmail.com)
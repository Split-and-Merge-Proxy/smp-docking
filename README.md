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
**Note:** You can change the `data_fraction` in the shell script to control the amount of training data used.

### SMP
```bash
bash ./scripts/smp_pretrain.sh
bash ./scripts/smp_finetune.sh
```
**Note:** You can change the `data_fraction` in the shell script to adjust the amount of data used for fine-tuning (pre-training currently does not support this option). You may also change the `resume_ckpt` field to specify the path to the pre-trained checkpoint.



## 4. Evaluations

```bash
bash ./scripts/test.sh
```
**Note:** You can change the `method_name` in `test.sh` to specify whether to evaluate the EquiDock or SMP method, and change the `ckpt_path` to point to the corresponding checkpoint.

## 5. Reproducing the Results Reported in the Manuscript
To reproduce the results reported in our manuscript, we provide the test PDB files in the `test_sets_pdb` folder. You can change the `method_name` in `test.sh` to specify whether to evaluate the EquiDock or SMP method. We also provide the corresponding checkpoints for both methods in the `./checkpts` folder. After downloading them, update the `ckpt_path` in `test.sh` to your local path and then run the following command:

```bash
bash ./scripts/test.sh
```

The expected results are shown below.

**DIPS-Het**
||   | Complex RMSD |  |  | Interface RMSD |  |  |
|----------|----------|----------|----------|----------|----------|----------|----------|
|Method| Median  | Mean | Std| Median | Mean | Std | Success Rate |
|EquiDock| 8.63 | 12.00 | 10.18  | 8.67 | 10.43 | 8.72 | 30% |
|SMP| 8.26  | 10.77  | 9.21  | 8.19 | 9.46 | 7.87 | 35%|

## 6. Infernce on Your Custom Data
We have uploaded the trained SMP weights in the `./ckpts` directory. You can download them and place them in your preferred location. Additionally, we provide an example in `./example` that demonstrates how to perform inference on your own custom data. You can run it with:

```bash
python -u custom_inference.py
```
The output will be saved in the `./save` directory as a pair of docked PDB files.

**Note:** Current deep learning–based docking methods may still struggle to produce satisfactory results. We recommend using such methods with caution.




## Acknowledges
- [EquiDock](https://github.com/octavian-ganea/equidock_public)
- [EBMDock](https://github.com/wuhuaijin/EBMDock)
- [HMR](https://github.com/bytedance/HMR)
- [DIPS](https://github.com/drorlab/DIPS)



If you have any questions, please don't hesitate to contact me through [cs.dh97@gmail.com](cs.dh97@gmail.com)
# python -m src.test_all_methods.eval_pdb_outputset

import os

from biopandas.pdb import PandasPdb
import numpy as np
from tqdm import tqdm

import torch

from src.utils.eval import Meter_Unbound_Bound
import scipy.spatial as spa


def get_CA_coords(pdb_file):
    ppdb_model = PandasPdb().read_pdb(pdb_file)
    df = ppdb_model.df['ATOM']
    df = df[df['atom_name'] == 'CA']
    return df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)


def compute_single_test_rmsd(input_dir, ground_truth_dir, file_names, method):

    meter = Meter_Unbound_Bound()

    Irmsd_meter = Meter_Unbound_Bound()
    dockq_meter = Meter_Unbound_Bound()
    fnat_meter = Meter_Unbound_Bound()

    all_crmsd = []
    all_irmsd = []
    all_dockq = []

    for name in tqdm(file_names):
        if method == 'geodock':
            ligand_model_file = os.path.join(input_dir, name + '_l_b.pdb')
            receptor_model_file = os.path.join(input_dir, name + '_r_b.pdb')
        elif method == 'diffdock':
            if name == 'FP_6FPG_E_BCD.pdb_0.dill':
                all_crmsd.append(0)
                all_irmsd.append(0)
                all_dockq.append(0)
                continue
            ll = len('.pdb_0.dill')
            ligand_model_file = os.path.join(input_dir, name[:-ll] + '_l_b_' + method.upper() + '.pdb')
            receptor_model_file = os.path.join(input_dir, name[:-ll] + '_r_b_' + method.upper() + '.pdb')
        else:
            ligand_model_file = os.path.join(input_dir, name + '_l_b_' + method.upper() + '.pdb')
            receptor_model_file = os.path.join(ground_truth_dir, name + '_r_b' + '_COMPLEX.pdb')

        ligand_gt_file = os.path.join(ground_truth_dir, name + '_l_b' + '_COMPLEX.pdb')
        receptor_gt_file = os.path.join(ground_truth_dir, name + '_r_b' + '_COMPLEX.pdb')


        ligand_model_coords = get_CA_coords(ligand_model_file)
        receptor_model_coords = get_CA_coords(receptor_model_file)

        ligand_gt_coords = get_CA_coords(ligand_gt_file)
        receptor_gt_coords = get_CA_coords(receptor_gt_file)

        assert ligand_model_coords.shape[0] == ligand_gt_coords.shape[0]
        assert receptor_model_coords.shape[0] == receptor_gt_coords.shape[0]

        ligand_receptor_distance = spa.distance.cdist(ligand_gt_coords, receptor_gt_coords)
        positive_tuple = np.where(ligand_receptor_distance < 8.)
        active_ligand = positive_tuple[0]
        active_receptor = positive_tuple[1]
        ligand_model_pocket_coors = ligand_model_coords[active_ligand, :]
        receptor_model_pocket_coors = receptor_model_coords[active_receptor, :]
        ligand_gt_pocket_coors = ligand_gt_coords[active_ligand, :]
        receptor_gt_pocket_coors = receptor_gt_coords[active_receptor, :]


        crmsd, ligand_rmsd = meter.update_rmsd(torch.Tensor(ligand_model_coords), torch.Tensor(receptor_model_coords),
                          torch.Tensor(ligand_gt_coords), torch.Tensor(receptor_gt_coords))

        irmsd, _ = Irmsd_meter.update_rmsd(torch.Tensor(ligand_model_pocket_coors), torch.Tensor(receptor_model_pocket_coors),
                                torch.Tensor(ligand_gt_pocket_coors), torch.Tensor(receptor_gt_pocket_coors))
        fnat = fnat_meter.update_Fnat(torch.Tensor(ligand_model_coords), torch.Tensor(receptor_model_coords),
                          torch.Tensor(ligand_gt_coords), torch.Tensor(receptor_gt_coords))
        dockq = dockq_meter.update_dockq(fnat, irmsd, ligand_rmsd)


        all_crmsd.append(crmsd)
        all_irmsd.append(irmsd)
        all_dockq.append(dockq)
    
    return all_crmsd, all_irmsd, all_dockq


def compute_all_test_rmsd(dataset):
    print('\n ' + dataset)

    geodock_input_dir = './test_sets_pdb/dips_het_geodock_results/'
    diffdock_input_dir = './test_sets_pdb/dips_het_diffdock_results/'
    equidock_input_dir = './test_sets_pdb/dips_het_equidock_results/'
    smp_input_dir = './test_sets_pdb/dips_het_smp_op_results/'

    ground_truth_dir = './test_sets_pdb/' + dataset + '_test_random_transformed/complexes/'

    file_names = []
    for file in os.listdir(geodock_input_dir):
        ll = len('_l_b' + '.pdb')
        file_names.append(file[:-ll])
    
    # print(file_names)

    geodock_crmsd, geodock_irmsd, geodock_dockq = compute_single_test_rmsd(geodock_input_dir, ground_truth_dir, file_names, method='geodock')
    diffdock_crmsd, diffdock_irmsd, diffdock_dockq = compute_single_test_rmsd(diffdock_input_dir, ground_truth_dir, file_names, method='diffdock')
    equidock_crmsd, equidock_irmsd, equidock_dockq = compute_single_test_rmsd(equidock_input_dir, ground_truth_dir, file_names, method='equidock')
    smp_crmsd, smp_irmsd, smp_dockq = compute_single_test_rmsd(smp_input_dir, ground_truth_dir, file_names, method='smp_op')

    results = []
    for g_dockq, d_dockq, e_dockq, s_dockq, file_name in zip(geodock_dockq, diffdock_dockq, equidock_dockq, smp_dockq, file_names):
        
        if np.isnan(g_dockq):
            g_dockq = 0
        if np.isnan(d_dockq):
            d_dockq = 0
        if np.isnan(e_dockq):
            e_dockq = 0
        if np.isnan(s_dockq):
            e_dockq = 0
        
        arr = np.array([g_dockq, d_dockq, e_dockq, s_dockq])
        index = np.argmax(arr)
        if index == 3 and s_dockq > 0.23:
            results.append([g_dockq, d_dockq, e_dockq, s_dockq, file_name])


    print(results)

## Run this to get the results from our paper.
if __name__ == "__main__":
    compute_all_test_rmsd('dips_het') # dataset: db5 or dips; methods: equidock, attract, patchdock, cluspro, hdock

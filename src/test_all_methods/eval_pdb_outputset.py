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


def compute_all_test_rmsd(dataset, method, data_frac=None):
    print('\n ' + dataset + ' ' + method)

    input_dir = './test_sets_pdb/' + dataset + '_' + method + '_results' + '_' + str(data_frac)
    ground_truth_dir = './test_sets_pdb/' + dataset + '_test_random_transformed/complexes/'


    pdb_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.pdb')]

    meter = Meter_Unbound_Bound()

    Irmsd_meter = Meter_Unbound_Bound()
    dockq_meter = Meter_Unbound_Bound()
    fnat_meter = Meter_Unbound_Bound()

    all_crmsd = []
    all_irmsd = []
    all_dockq = []
    rate = 0

    num_test_files = 0
    all_files = []
    for file in tqdm(pdb_files):

        ######## ATTRACT EVAL ###########
        if method == 'attract':
            if not file.endswith('_l_b_ATTRACT.pdb'):
                continue
            ll = len('_l_b_ATTRACT.pdb')
            ligand_model_file = os.path.join(input_dir, file[:-ll] + '_l_b' + '_ATTRACT.pdb')
            ligand_gt_file = os.path.join(ground_truth_dir, file[:-ll] + '_l_b' + '_COMPLEX.pdb')
            receptor_model_file = os.path.join(input_dir, file[:-ll] + '_r_b' + '_ATTRACT.pdb')
            receptor_gt_file = os.path.join(ground_truth_dir, file[:-ll] + '_r_b' + '_COMPLEX.pdb')
        #################################

        elif method == 'hdock':
            if not file.endswith('_l_b_HDOCK.pdb'):
                continue
            ll = len('_l_b_HDOCK.pdb')
            ligand_model_file = os.path.join(input_dir, file[:-ll] + '_l_b' + '_HDOCK.pdb')
            receptor_model_file = os.path.join(input_dir, file[:-ll] + '_r_b' + '_HDOCK.pdb')
            if dataset == 'db5':
                ligand_gt_file = os.path.join(ground_truth_dir, file[:-ll] + '_l_b' + '_COMPLEX.pdb')
                receptor_gt_file = os.path.join(ground_truth_dir, file[:-ll] + '_r_b' + '_COMPLEX.pdb')
            elif dataset == 'dips':
                ligand_gt_file = os.path.join(ground_truth_dir, file[:-ll] + '.pdb_0.dill_l_b' + '_COMPLEX.pdb')
                receptor_gt_file = os.path.join(ground_truth_dir, file[:-ll] + '.pdb_0.dill_r_b' + '_COMPLEX.pdb')

        #################################
        elif method == 'geodock' or method == 'patchdock':
            if not file.endswith('_l_b' + '.pdb'):
                continue
            ll = len('_l_b' + '.pdb')
            ligand_model_file = os.path.join(input_dir, file[:-ll] + '_l_b' + '.pdb')
            ligand_gt_file = os.path.join(ground_truth_dir, file[:-ll] + '_l_b' + '_COMPLEX.pdb')
            receptor_model_file = os.path.join(input_dir, file[:-ll] + '_r_b' + '.pdb')
            receptor_gt_file = os.path.join(ground_truth_dir, file[:-ll] + '_r_b' + '_COMPLEX.pdb')
        else:

            if not file.endswith('_l_b_' + method.upper() + '.pdb'):
                continue
            # if not file.endswith('_l_b' + '.pdb'):
            #     continue
            ll = len('_l_b_' + method.upper() + '.pdb')
            # ll = len('_l_b' + '.pdb')
            ligand_model_file = os.path.join(input_dir, file[:-ll] + '_l_b_' + method.upper() + '.pdb')
            # ligand_model_file = os.path.join(input_dir, file[:-ll] + '_l_b' + '.pdb')
            # ligand_model_file = os.path.join(ground_truth_dir, file[:-ll] + '_l_b' + '_COMPLEX.pdb')
            ligand_gt_file = os.path.join(ground_truth_dir, file[:-ll] + '_l_b' + '_COMPLEX.pdb')
            receptor_model_file = os.path.join(ground_truth_dir, file[:-ll] + '_r_b' + '_COMPLEX.pdb')
            # receptor_model_file = os.path.join(input_dir, file[:-ll] + '_r_b' + '.pdb')
            receptor_gt_file = os.path.join(ground_truth_dir, file[:-ll] + '_r_b' + '_COMPLEX.pdb')

        num_test_files += 1

        ligand_model_coords = get_CA_coords(ligand_model_file)
        receptor_model_coords = get_CA_coords(receptor_model_file)

        ligand_gt_coords = get_CA_coords(ligand_gt_file)
        receptor_gt_coords = get_CA_coords(receptor_gt_file)

        # print(ligand_gt_coords.shape, ligand_model_coords.shape)
        # print(ligand_model_file)

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
        
        """
        if crmsd > 50:
            # remove the sample if the crmsd is too large
            num_test_files -= 1
            continue
        """
        irmsd, _ = Irmsd_meter.update_rmsd(torch.Tensor(ligand_model_pocket_coors), torch.Tensor(receptor_model_pocket_coors),
                                torch.Tensor(ligand_gt_pocket_coors), torch.Tensor(receptor_gt_pocket_coors))
        fnat = fnat_meter.update_Fnat(torch.Tensor(ligand_model_coords), torch.Tensor(receptor_model_coords),
                          torch.Tensor(ligand_gt_coords), torch.Tensor(receptor_gt_coords))
        dockq = dockq_meter.update_dockq(fnat, irmsd, ligand_rmsd)

        if dockq >= 0.23 and not np.isnan(dockq):
            rate += 1

        all_crmsd.append(crmsd)
        all_irmsd.append(irmsd)
        all_dockq.append(dockq)
        all_files.append(file)

    print('crmsd = ', str(all_crmsd))
    print('irmsd = ', str(all_irmsd))
    print('dockq = ', str(all_dockq))
    print('success rate:', rate / num_test_files)
    # print('all file:', all_files)


    complex_rmsd_median, _ = meter.summarize_with_std(reduction_rmsd='median')
    complex_interface_rmsd_median, _ = Irmsd_meter.summarize_with_std(reduction_rmsd='median')
    complex_rmsd_mean, complex_rmsd_std = meter.summarize_with_std(reduction_rmsd='mean')
    complex_interface_rmsd_mean, complex_interface_rmsd_std = Irmsd_meter.summarize_with_std(reduction_rmsd='mean')

    dockq_median, _ = dockq_meter.summarize_dockq_with_std(reduction_rmsd='median')
    dockq_mean, dockq_std = dockq_meter.summarize_dockq_with_std(reduction_rmsd='mean')

    print('For ', dataset, ' method = ', method, '; num test files =',  num_test_files,
          ' complex_rmsd_CA median/mean/std = ', complex_rmsd_median, '/', complex_rmsd_mean, ' +- ', complex_rmsd_std,
          ' complex_interface_rmsd CA median/mean/std = ', complex_interface_rmsd_median, '/', complex_interface_rmsd_mean, ' +- ', complex_interface_rmsd_std,
          'dockq median/mean/std = ', dockq_median, '/', dockq_mean, ' +- ', dockq_std)


## Run this to get the results from our paper.
if __name__ == "__main__":
    compute_all_test_rmsd('dips_het', 'smp', 0.2) # dataset: db5 or dips; methods: equidock, attract, patchdock, cluspro, hdock

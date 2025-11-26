import os
import torch

os.environ['DGLBACKEND'] = 'pytorch'

from src.utils.protein_utils import preprocess_unbound_bound, protein_to_graph_unbound_bound
from biopandas.pdb import PandasPdb
from src.utils.train_utils import *
from src.utils.args import *
from src.utils.ot_utils import *
from src.utils.zero_copy_from_numpy import *
from src.utils.io import create_dir


def get_nodes_coors_numpy(filename, all_atoms=False):
    df = PandasPdb().read_pdb(filename).df['ATOM']
    if not all_atoms:
        return torch.from_numpy(df[df['atom_name'] == 'CA'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32))
    return torch.from_numpy(df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32))

def get_residues(pdb_filename):
    df = PandasPdb().read_pdb(pdb_filename).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    return residues


def main():

    ligand_file_path = './example/EE_4EEC_B_AC.pdb_0.dill_l_b.pdb'
    receptor_file_path = './example/EE_4EEC_B_AC.pdb_0.dill_r_b.pdb'
    ckpt_file_path = './checkpts/smp/dips_het_model_best.pth'
    remove_clashes = False
    output_dir = './save'

    ckpt = torch.load(ckpt_file_path, map_location=args['device'])
    model = create_model(args, log)
    model.load_state_dict(checkpoint['state_dict'])
    param_count(model, log)
    model = model.to(args['device'])
    model.eval()

    print(' inference on file = ', ligand_file_path)

    ppdb_ligand = PandasPdb().read_pdb(ligand_file_path)
    ppdb_receptor = PandasPdb().read_pdb(receptor_file_path)
    unbound_ligand_all_atoms_pre_pos = ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)

    initial_ligand_nodes_coors = get_nodes_coors_numpy(ligand_file_path, all_atoms=True)
    unbound_predic_ligand, \
    unbound_predic_receptor, \
    bound_ligand_repres_nodes_loc_clean_array,\
    bound_receptor_repres_nodes_loc_clean_array = preprocess_unbound_bound(
        get_residues(ligand_filename), get_residues(receptor_filename),
        graph_nodes=args['graph_nodes'], pos_cutoff=args['pocket_cutoff'], inference=True)

    ligand_graph, receptor_graph = protein_to_graph_unbound_bound(unbound_predic_ligand,
                                                                    unbound_predic_receptor,
                                                                    bound_ligand_repres_nodes_loc_clean_array,
                                                                    bound_receptor_repres_nodes_loc_clean_array,
                                                                    graph_nodes=args['graph_nodes'],
                                                                    cutoff=args['graph_cutoff'],
                                                                    max_neighbor=args['graph_max_neighbor'],
                                                                    one_hot=False,
                                                                    residue_loc_is_alphaC=args['graph_residue_loc_is_alphaC']
                                                                    )

    if args['input_edge_feats_dim'] < 0:
        args['input_edge_feats_dim'] = ligand_graph.edata['he'].shape[1]


    ligand_graph.ndata['new_x'] = ligand_graph.ndata['x']

    assert np.linalg.norm(bound_ligand_repres_nodes_loc_clean_array - ligand_graph.ndata['x'].detach().cpu().numpy()) < 1e-1

    # Create a batch of a single DGL graph
    batch_hetero_graph = batchify_and_create_hetero_graphs_inference(ligand_graph, receptor_graph)

    batch_hetero_graph = batch_hetero_graph.to(args['device'])
    model_ligand_coors_deform_list, \
    model_keypts_ligand_list, model_keypts_receptor_list, \
    all_rotation_list, all_translation_list = model(batch_hetero_graph, epoch=0)


    rotation = all_rotation_list[0].detach().cpu().numpy()
    translation = all_translation_list[0].detach().cpu().numpy()

    new_residues = (rotation @ bound_ligand_repres_nodes_loc_clean_array.T).T+translation
    # assert np.linalg.norm(new_residues - model_ligand_coors_deform_list[0].detach().cpu().numpy()) < 1e-1

    unbound_ligand_new_pos = (rotation @ unbound_ligand_all_atoms_pre_pos.T).T+translation

    euler_angles_finetune = torch.zeros([3], requires_grad=True)
    translation_finetune = torch.zeros([3], requires_grad=True)
    ligand_th = (get_rot_mat(euler_angles_finetune) @ torch.from_numpy(unbound_ligand_new_pos).T).T + translation_finetune

    ## Optimize the non-intersection loss:
    if remove_clashes:
        non_int_loss_item = 100.
        it = 0
        while non_int_loss_item > 0.5 and it < 2000:
            non_int_loss = compute_body_intersection_loss(ligand_th, gt_receptor_nodes_coors, sigma=8, surface_ct=8)
            non_int_loss_item = non_int_loss.item()
            eta = 1e-3
            if non_int_loss < 2.:
                eta = 1e-4
            if it > 1500:
                eta = 1e-2
            if it % 100 == 0:
                print(it, ' ' , non_int_loss_item)
            non_int_loss.backward()
            translation_finetune = translation_finetune - eta * translation_finetune.grad.detach()
            translation_finetune = torch.tensor(translation_finetune, requires_grad=True)

            euler_angles_finetune = euler_angles_finetune - eta * euler_angles_finetune.grad.detach()
            euler_angles_finetune = torch.tensor(euler_angles_finetune, requires_grad=True)

            ligand_th = (get_rot_mat(euler_angles_finetune) @ torch.from_numpy(unbound_ligand_new_pos).T).T + translation_finetune

            it += 1


    ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = ligand_th.detach().numpy() # unbound_ligand_new_pos
    unbound_ligand_save_file_path = os.path.join(output_dir, 'ligand.pdb')
    unbound_receptor_save_file_path = os.path.join(output_dir, 'receptor.pdb')
    ppdb_ligand.to_pdb(path=unbound_ligand_save_filename, records=['ATOM'], gz=False)
    ppdb_receptor.to_pdb(path=unbound_receptor_save_file_path, records=['ATOM'], gz=False)


if __name__ == "__main__":
    main()
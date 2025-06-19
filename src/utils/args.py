# Ignore future warning
import sys
import warnings
import datetime
import random

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import os
import torch

print('Parsing args')

parser = argparse.ArgumentParser(description='Docking')


parser.add_argument('-debug', default=False, action='store_true')

parser.add_argument('-log_every', default=100000, type=int, required=False, help='log frequency during training')
parser.add_argument('-random_seed', type=int, required=False, default=8, help='random seed')

# Method
parser.add_argument('-method', type=str, required=False, default='equidock', choices=['equidock', 'smp'])
parser.add_argument('-resume_ckpt', type=str, required=False, default=None, help='resume checkpoint')

# Data
parser.add_argument('-data', type=str, required=False, default='dips_het', choices=['dips_het', 'pseudo_multi'])
parser.add_argument('-data_fraction', type=float, default=1., required=False)
parser.add_argument('-split', type=int, required=False, default=0, help='cross valid split')
parser.add_argument('-worker', type=int, default=5, required=False, help="Number of worker for data loader.")
parser.add_argument('-n_jobs', type=int, default=10, required=False, help="Number of worker for data preprocessing")


# Optim and Scheduler
parser.add_argument('-lr', type=float, default=3e-4, required=False)
parser.add_argument('-w_decay', type=float, default=1e-4, required=False)
parser.add_argument('-scheduler', default='warmup', choices=['ROP', 'warmup', 'cyclic'])
parser.add_argument('-warmup', type=float, default=1., required=False)
parser.add_argument('-patience', type=int, default=50, required=False, help='patience')
parser.add_argument('-num_epochs', type=int, default=10000, required=False, help="Used when splitting data for horovod.")
parser.add_argument('-clip', type=float, default=100., required=False, help="Gradient clip threshold.")
parser.add_argument('-bs', type=int, default=10, required=False)


### GRAPH characteristics and features
parser.add_argument('-graph_nodes', type=str, default='residues', required=False, choices=['residues'])
#################### Only for data caching, inference and to know which data to load.
parser.add_argument('-graph_cutoff', type=float, default=30., required=False, help='Only for data caching and inference.')
parser.add_argument('-graph_max_neighbor', type=int, default=10, required=False, help='Only for data caching and inference.')
parser.add_argument('-graph_residue_loc_is_alphaC', default=False, action='store_true',
                    help='whether to use coordinates of alphaC or avg of atom locations as the representative residue location.'
                         'Only for data caching and inference.')
parser.add_argument('-pocket_cutoff', type=float, default=8., required=False)


# Unbound - bound initial positions
parser.add_argument('-translation_interval', default=5.0, type=float, required=False, help='translation interval')

# Model
parser.add_argument('-rot_model', type=str, default='kb_att', choices=['kb_att'])
parser.add_argument('-num_att_heads', type=int, default=50, required=False)



## Pocket OT:
parser.add_argument('-pocket_ot_loss_weight', type=float, default=1., required=False)


# Intersection loss:
parser.add_argument('-intersection_loss_weight', type=float, default=10., required=False)
parser.add_argument('-intersection_sigma', type=float, default=25., required=False)
parser.add_argument('-intersection_surface_ct', type=float, default=10., required=False)

parser.add_argument('-dropout', type=float, default=0., required=False)
parser.add_argument('-layer_norm', type=str, default='LN', choices=['0', 'BN', 'LN'])
parser.add_argument('-layer_norm_coors', type=str, default='0', choices=['0', 'LN'])
parser.add_argument('-final_h_layer_norm', type=str, default='0', choices=['0', 'GN', 'BN', 'LN'])

parser.add_argument('-nonlin', type=str, default='lkyrelu', choices=['swish', 'lkyrelu'])
parser.add_argument('-iegmn_lay_hid_dim', type=int, default=64, required=False)
parser.add_argument('-iegmn_n_lays', type=int, default=8, required=False)
parser.add_argument('-residue_emb_dim', type=int, default=64, required=False, help='embedding')
parser.add_argument('-shared_layers', default=False, action='store_true')
parser.add_argument('-cross_msgs', default=False, action='store_true')

parser.add_argument('-divide_coors_dist', default=False, action='store_true')


parser.add_argument('-use_dist_in_layers', default=False, action='store_true')
parser.add_argument('-use_edge_features_in_gmn', default=False, action='store_true')

parser.add_argument('-noise_decay_rate', type=float, default=0., required=False)
parser.add_argument('-noise_initial', type=float, default=0., required=False)

parser.add_argument('-use_mean_node_features', default=False, action='store_true')

parser.add_argument('-skip_weight_h', type=float, default=0.75, required=False)

parser.add_argument('-leakyrelu_neg_slope', type=float, default=1e-2, required=False)


parser.add_argument('-x_connection_init', type=float, default=0., required=False)

## Hyper search
# parser.add_argument('-hyper_search', default=False, action='store_true')


parser.add_argument('-fine_tune', default=False, action='store_true') ## Some fine-tuning E-GNN model that didn't work, feel free to play with it.


# parser.add_argument('-toy', default=False, action='store_true') ## Train only on DB5.5


# parser.add_argument('-continue_train_model', type=str, default='')


args = parser.parse_args().__dict__


args['device'] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Available GPUS:{torch.cuda.device_count()}")

if torch.cuda.is_available():
    torch.cuda.set_device(0)


########################################
def get_model_name(args):

    method = args['method']
    data_name = args['data']
    if method == 'smp':
        if args['resume_ckpt'] is None:
            model_name = method + '_pretrain' + '_' + data_name
        else:
            model_name = method + '_finetune' + '_' + data_name + '_' + str(args['data_fraction'])
    else:
        model_name = method + '_' + data_name + '_' + str(args['data_fraction'])

    assert len(model_name) <= 255
    return model_name
########################################


assert args['noise_decay_rate'] < 1., 'Noise has to decrease to 0, decay rate cannot be >= 1.'



banner = get_model_name(args)

print(banner)


def pprint(*kargs):
    print('[' + str(datetime.datetime.now()) + '] ', *kargs)

def log(*pargs): # redefined for train
    pprint(*pargs)

log('Model name ===> ', banner)

args['cache_path'] = './cache/' + args['data'] + '_' + args['graph_nodes'] + '_maxneighbor_' + \
                     str(args['graph_max_neighbor']) + '_cutoff_' + str(args['graph_cutoff']) + \
                     '_pocketCut_' + str(args['pocket_cutoff']) + '/'
args['cache_path'] = os.path.join(args['cache_path'], 'cv_' + str(args['split']))

args['checkpoint_dir'] = './checkpts/' + banner
args['checkpoint_filename'] = os.path.join(args['checkpoint_dir'], args['data'] + '_model_best.pth')
args['tb_log_dir'] = './tb_logs/'  + banner


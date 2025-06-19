set -x


CUDA_VISIBLE_DEVICES=0 python -m src.train -graph_residue_loc_is_alphaC -cross_msgs -use_dist_in_layers \
                                -use_edge_features_in_gmn -use_mean_node_features \
                                -lr 3e-4 -data 'dips_het' -data_fraction 1.0 -patience 30 -warmup 1.0 -split 0 -method equidock
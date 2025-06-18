# -*- coding: utf-8 -*-
#
# Evaluation of model performance."""
# pylint: disable= no-member, arguments-differ, invalid-name

import numpy as np
from sklearn.neighbors import BallTree

from src.utils.protein_utils import rigid_transform_Kabsch_3D

__all__ = ['Meter_Unbound_Bound']

class Meter_Unbound_Bound(object):
  def __init__(self):
    self.complex_rmsd_list = []
    self.ligand_rmsd_list = []
    self.receptor_rmsd_list = []
    self.dockq_list = []
    self.fnat_list = []


  def update_rmsd(self, ligand_coors_pred, receptor_coors_pred, ligand_coors_true, receptor_coors_true):

    ligand_coors_pred = ligand_coors_pred.detach().cpu().numpy()
    receptor_coors_pred = receptor_coors_pred.detach().cpu().numpy()

    ligand_coors_true = ligand_coors_true.detach().cpu().numpy()
    receptor_coors_true = receptor_coors_true.detach().cpu().numpy()

    ligand_rmsd = np.sqrt(np.mean(np.sum( (ligand_coors_pred - ligand_coors_true) ** 2, axis=1)))
    receptor_rmsd = np.sqrt(np.mean(np.sum( (receptor_coors_pred - receptor_coors_true) ** 2, axis=1)))

    complex_coors_pred = np.concatenate((ligand_coors_pred, receptor_coors_pred), axis=0)
    complex_coors_true = np.concatenate((ligand_coors_true, receptor_coors_true), axis=0)

    R,b = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
    complex_coors_pred_aligned = ( (R @ complex_coors_pred.T) + b ).T

    complex_rmsd = np.sqrt(np.mean(np.sum( (complex_coors_pred_aligned - complex_coors_true) ** 2, axis=1)))

    if not np.isnan(complex_rmsd):
      self.complex_rmsd_list.append(complex_rmsd)
      self.ligand_rmsd_list.append(ligand_rmsd)
      self.receptor_rmsd_list.append(receptor_rmsd)

    return complex_rmsd, ligand_rmsd


  def summarize(self, reduction_rmsd='median'):
    if reduction_rmsd == 'mean':
      complex_rmsd_array = np.array(self.complex_rmsd_list)
      complex_rmsd_summarized = np.mean(complex_rmsd_array)

      ligand_rmsd_array = np.array(self.ligand_rmsd_list)
      ligand_rmsd_summarized = np.mean(ligand_rmsd_array)

      receptor_rmsd_array = np.array(self.receptor_rmsd_list)
      receptor_rmsd_summarized = np.mean(receptor_rmsd_array)
    elif reduction_rmsd == 'median':
      complex_rmsd_array = np.array(self.complex_rmsd_list)
      complex_rmsd_summarized = np.median(complex_rmsd_array)

      ligand_rmsd_array = np.array(self.ligand_rmsd_list)
      ligand_rmsd_summarized = np.median(ligand_rmsd_array)

      receptor_rmsd_array = np.array(self.receptor_rmsd_list)
      receptor_rmsd_summarized = np.median(receptor_rmsd_array)
    else:
      raise ValueError("Meter_Unbound_Bound: reduction_rmsd mis specified!")
    return ligand_rmsd_summarized, receptor_rmsd_summarized, complex_rmsd_summarized


  def summarize_with_std(self, reduction_rmsd='median'):
    complex_rmsd_array = np.array(self.complex_rmsd_list)
    if reduction_rmsd == 'mean':
      complex_rmsd_summarized = np.mean(complex_rmsd_array)
    elif reduction_rmsd == 'median':
      complex_rmsd_summarized = np.median(complex_rmsd_array)
    else:
      raise ValueError("Meter_Unbound_Bound: reduction_rmsd mis specified!")
    return complex_rmsd_summarized, np.std(complex_rmsd_array)


  def update_Fnat(self, lig, rec, lig_gt, rec_gt, iface_cutoff=8.0):
    bt1_gt = BallTree(lig_gt)
    dist1_gt, _ = bt1_gt.query(rec_gt, k=1)
    rec_iface_ind_gt = np.where(dist1_gt < iface_cutoff)[0]
    if len(rec_iface_ind_gt) == 0:
      return 0.0
    lig_iface_ind_gt = bt1_gt.query_radius(rec_gt[rec_iface_ind_gt], iface_cutoff)
    iface_pair_gt = set([(i,j) for n, i in enumerate(rec_iface_ind_gt) for j in lig_iface_ind_gt[n]])

    bt1 = BallTree(lig)
    dist1, _ = bt1.query(rec, k=1)
    rec_iface_ind = np.where(dist1 < iface_cutoff)[0]
    if len(rec_iface_ind) == 0:
      return 0.0
    
    lig_iface_ind = bt1.query_radius(rec[rec_iface_ind], iface_cutoff)
    iface_pair = set([(i,j) for n, i in enumerate(rec_iface_ind) for j in lig_iface_ind[n]])

    return len(iface_pair & iface_pair_gt) / len(iface_pair_gt)

  def update_dockq(self, Fnat, IRMSD, LRMSD):
    # import pdb; pdb.set_trace()
    dockq = ( Fnat + 1.0 / (1.0 + (IRMSD/1.5)**2) + 1.0 / (1.0 + (LRMSD/8.5)**2) )/ 3.0
    if not np.isnan(dockq):
      self.dockq_list.append(dockq)
    
    return dockq

  def summarize_dockq_with_std(self, reduction_rmsd='median'):
    dockq_array = np.array(self.dockq_list)
    if reduction_rmsd == 'mean':
      dockq_summarized = np.mean(dockq_array)
    elif reduction_rmsd == 'median':
      dockq_summarized = np.median(dockq_array)
    else:
      raise ValueError("Meter_Unbound_Bound: reduction_rmsd mis specified!")
    return dockq_summarized, np.std(dockq_array)
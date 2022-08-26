import math
from typing import List, Optional, Union

import torch


def num_swapped_pairs(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> int:
    order = ys_true.argsort(descending=True)
    pairs = torch.combinations(order, 2)
    mask_true_equal = ys_true[pairs[:, 0]] != ys_true[pairs[:, 1]]
    mask_pred = ys_pred[pairs[:, 0]] < ys_pred[pairs[:, 1]]
    return (mask_true_equal & mask_pred).sum().item()


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'exp2':
        return 2**y_value - 1.0
    return y_value


def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str) -> float:    
    order = ys_pred.argsort(descending=True)
    index = torch.arange(len(order), dtype=torch.float64) + 1
    return (compute_gain(ys_true[order], gain_scheme) / torch.log2(index + 1)).sum().item()
    

def ndcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str = 'const') -> float:
    dcg_val = dcg(ys_true, ys_pred, gain_scheme)
    dcg_best_val = dcg(ys_true, ys_true, gain_scheme)
    return dcg_val / dcg_best_val


def precission_at_k(ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
    total_relevant = ys_true.sum().item()
    if total_relevant == 0:
        return -1
    
    order = ys_pred.argsort(descending=True)[:k]
    n_retrieved = len(order)
    n_relevant = (ys_true[order] == 1).sum().item()
    if n_retrieved > total_relevant:
        return n_relevant / total_relevant
    else:
        return n_relevant / n_retrieved


def reciprocal_rank(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    order = ys_pred.argsort(descending=True)
    return 1 / (ys_true[order].argsort(descending=True)[0] + 1)


def p_found(ys_true: torch.Tensor, ys_pred: torch.Tensor, p_break: float = 0.15 ) -> float:
    order = ys_pred.argsort(descending=True)
    
    p_rels = ys_true[order]
    p_look_ = 1
    p_rel_ = p_rels[0].item()
    p_found = p_look_ * p_rel_
    for i in range(1, len(ys_true)):
        p_rel = p_rels[i].item()
        p_look = p_look_ * (1 - p_rel_) * (1 - p_break)
        
        p_found += p_look * p_rel
        
        p_rel_ = p_rel
        p_look_ = p_look
    
    return p_found
    


def average_precision(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    if ys_true.sum() == 0:
        return -1
    
    order = ys_pred.argsort(descending=True)
    
    n = ys_true.sum().item()
    recall_ = 0.0
    ap = 0.0
    for k in range(1, len(ys_true) + 1):        
        n_relevant = (ys_true[order][:k] == 1).sum().item()
        
        precision = n_relevant / k
        recall = n_relevant / n
        
        ap += (recall - recall_) * precision
        recall_ = recall
    
    return ap

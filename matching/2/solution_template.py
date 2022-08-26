import math
from typing import List, Optional, Union

import torch


def num_swapped_pairs(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> int:
    pass


def compute_gain(y_value: float, gain_scheme: str) -> float:
    pass


def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str) -> float:
    pass


def ndcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str = 'const') -> float:
    pass


def precission_at_k(ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
    pass


def reciprocal_rank(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    pass


def p_found(ys_true: torch.Tensor, ys_pred: torch.Tensor, p_break: float = 0.15 ) -> float:
    pass


def average_precision(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    pass

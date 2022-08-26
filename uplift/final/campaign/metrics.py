import numpy as np


def uplift_at_k(uplift: np.ndarray,
                w: np.ndarray,
                y: np.ndarray,
                k: float = 0.3) -> float:
    order = np.argsort(uplift)[::-1]

    n = int(len(order) * k)

    yt = y[order][w[order] == 1][:n].mean()
    yc = y[order][w[order] == 0][:n].mean()

    return yt - yc

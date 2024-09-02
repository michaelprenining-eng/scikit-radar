import dataclasses
from enum import Enum, auto

import numpy as np


class CFARMode(Enum):
    CA = auto()
    CAGO = auto()


@dataclasses.dataclass(frozen=True)
class CFARConfig:
    mode: CFARMode = CFARMode.CA
    train_cells: int = 3
    gard_cells: int = 0
    pfa: float = 1e-4


DEFAULT_CFG = CFARConfig()


def cfar_threshold(sig: np.ndarray, cfg: CFARConfig = DEFAULT_CFG):
    """Calculates the threshold level for a signal x with a CFAR method given by mode, by looping over all cells.

    Args:
        sig: array of positive (absolute values) of floats
        cfg: CFAR configuration

    Returns:
        array of size of x holding threshold levels
    """
    scale = 1 / cfg.train_cells

    kernel = np.full(cfg.train_cells, scale)
    corr = np.correlate(sig, kernel)

    offset = cfg.train_cells + 2 * cfg.gard_cells + 1
    left = corr[:-offset]
    right = corr[offset:]

    match cfg.mode:
        case CFARMode.CA:
            corr = 0.5 * (left + right)
        case CFARMode.CAGO:
            corr = np.maximum(left, right)
        case _:
            raise ValueError(f"CFAR mode '{str(cfg.mode)}' is not supported")

    nw2 = 2 * cfg.train_cells
    alpha = nw2 * (cfg.pfa ** (-1 / nw2) - 1)
    threshold = alpha * corr
    return threshold

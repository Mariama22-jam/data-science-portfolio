"""polymc: lattice Monte Carlo for self-avoiding polymer chains near a surface."""
from .core import (
    SYMMETRIES,
    gyration,
    is_connected,
    is_self_avoiding,
    n_surface,
    simulate_free_chain,
    simulate_grafted_chain,
    simulate_grafted_tempering,
)
from .analysis import blocking_error, fit_flory_exponent

__all__ = [
    "SYMMETRIES",
    "gyration",
    "is_connected",
    "is_self_avoiding",
    "n_surface",
    "simulate_free_chain",
    "simulate_grafted_chain",
    "simulate_grafted_tempering",
    "blocking_error",
    "fit_flory_exponent",
]
__version__ = "0.1.0"

r"""
Soft Analytical Side-Channel Attack (SASCA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: scalib.attacks

.. autosummary::
   :toctree:
   :recursive:
   :nosignatures:

   FactorGraph
   GenFactor
   BPState
"""

__all__ = [
        "FactorGraph", 
        "BPState", 
        "GenFactor",
        "OnlineCorrVector"
        ]

from .factor_graph import FactorGraph, BPState, GenFactor

from .pearson_corr import OnlineCorrVector

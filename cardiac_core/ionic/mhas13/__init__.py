"""
MHAS13 — Matured hiPSC-CM Ionic Model

Matured variant of the PHAS13 (Paci-Hyttinen-Aalto-Setala-Severi 2013)
model. Two modifications to suppress automaticity and produce a
quiescent, paced-only cell:

1. IK1 replaced with ten Tusscher (TTP06) formulation at GK1_critical
   (Fabbri et al. 2019, Biophys J 117:2303-15, PMID 31623886)
2. If (funny current) removed (g_f = 0)

Retains all other hiPSC-CM characteristics: ICaL GHK, Ca-dependent IKs,
smaller INa, immature Ca handling.

References:
- Paci M et al. (2013). Ann Biomed Eng 41(11):2334-2348.
- Fabbri A et al. (2019). Biophys J 117:2303-2315.
"""

from .model import MHAS13Model
from .parameters import MHAS13Parameters

__all__ = [
    'MHAS13Model',
    'MHAS13Parameters',
]

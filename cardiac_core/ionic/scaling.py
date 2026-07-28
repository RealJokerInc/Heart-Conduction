"""Shared ionic-conductance scaling — one validated home for the drug/upregulation knob.

Used by BOTH the tissue path (``api.CardiacSimulation.scale_conductance``) and the 0-D path
(``single_cell(conductances=...)``). Lives in ``ionic/`` (not ``api``) so the light 0-D driver can
import it without dragging in the engine solver stack.
"""
import copy


def scale_ionic_conductances(model, scalings):
    """Deep-copy ``model`` and multiply the named conductances on the copy.

    Uniform across models (TTP06/ORd/PHAS13/…): all expose their maximal conductances/
    permeabilities as attributes on ``self.params``. Operating on a deep copy of the LIVE
    engine model (not a freshly-named build) keeps the cell type and any prior scalings
    consistent across engines — the bidomain/LBM factories build ENDO by default, so
    re-deriving from name+mesh-cell_type would silently flip cell type. Raises
    ``ValueError`` on an unknown conductance name.
    """
    model = copy.deepcopy(model)
    params = model.params
    # Scalable = a true maximal conductance / permeability / transporter rate: an ohmic
    # `G*` (TTP06/ORd) or `g_*` (hiPSC paci/phas13/mhas13, LOWERCASE) or a `P*`
    # permeability/pump. EXCLUDES the `*_scale` tuning factors and the dimensionless
    # parameters that merely start with g/p — `gamma_ncx` (NCX voltage-partition) and
    # `PkNa` (the IKs permeability RATIO in the Nernst term) — which scale a shape/reversal,
    # not a magnitude, and which a bare `hasattr`/first-letter check would silently corrupt.
    _NON_CONDUCTANCE = {'gamma_ncx', 'pkna'}
    conductances = {
        a for a in vars(params)
        if a[:1].lower() in ('g', 'p')
        and not a.endswith('_scale')
        and a.lower() not in _NON_CONDUCTANCE
    }
    for name, factor in scalings.items():
        if name not in conductances:
            raise ValueError(
                f"{name!r} is not a scalable conductance of {type(model).__name__}; "
                f"available conductances: {sorted(conductances)}"
            )
        setattr(params, name, getattr(params, name) * float(factor))
    return model

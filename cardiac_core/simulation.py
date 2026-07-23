"""
Simulation — the engine-agnostic runtime Protocol.

A ``typing.Protocol`` (``runtime_checkable``) describing what every cardiac simulation exposes,
regardless of engine. Downstream tooling (parameter fitting, surrogate training, scripted
front-ends) programs against THIS, not against a concrete engine; the engine stays hidden behind
the ``monodomain()/bidomain()/lbm()`` factories.

``CardiacSimulation`` (in ``api.py``) satisfies this Protocol structurally — there is no inheritance.
``isinstance(sim, Simulation)`` is True because all members below are present.
"""

from typing import Iterator, Protocol, runtime_checkable

import torch


@runtime_checkable
class Simulation(Protocol):
    """Engine-agnostic simulation interface (introspection + run/step/reset/with_/stimulate)."""

    # --- introspection ---
    @property
    def Nx(self) -> int: ...
    @property
    def Ny(self) -> int: ...
    @property
    def dx(self) -> float: ...
    @property
    def dy(self) -> float: ...
    @property
    def dt(self) -> float: ...
    @property
    def Cm(self) -> float: ...
    @property
    def ionic_model(self) -> str: ...
    @property
    def engine_type(self) -> str: ...

    # --- runtime state ---
    @property
    def Vm(self) -> torch.Tensor:
        """Current membrane potential as an ``(Nx, Ny)`` grid."""
        ...

    @property
    def t(self) -> float: ...

    # --- control ---
    def run(self, t_end: float, save_every: float = 1.0) -> Iterator: ...
    def step(self) -> None: ...
    def reset(self) -> None: ...
    def with_(self, **overrides) -> "Simulation": ...
    def stimulate(self, region, start_time: float = 0.0, duration: float = 1.0,
                  amplitude: float = -52.0) -> None: ...

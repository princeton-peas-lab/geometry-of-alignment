"""Register typed schemas with Hydra's ConfigStore.

The top-level ``Config`` dataclass is the *primary* config (registered as
``config``). It carries its own ``defaults`` list selecting the experiment /
dataset / model groups, so CLIs use ``config_name="config"`` and merged group
YAMLs are validated against the typed group fields.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from .schema import BuildProjectedFisherCfg, BuildSubspaceCfg, Config


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    cs.store(name="build_subspace", node=BuildSubspaceCfg)
    cs.store(name="build_projected_fisher", node=BuildProjectedFisherCfg)

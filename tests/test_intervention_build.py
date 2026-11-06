"""Unit tests for FIM builder math + subspaces.pt load (no model load)."""

import torch
from omegaconf import OmegaConf

from interventions.base import load_subspaces_by_module
from interventions.build.repr import _compute_subspace_from_fim


def test_compute_subspace_from_fim_returns_top_eigenvectors():
    # Diagonal FIM with descending eigenvalues; top-2 eigenvectors are e0, e1.
    fim = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    eigvals, top = _compute_subspace_from_fim(fim, top_k=2)
    assert top.shape == (2, 4)  # (k, d), row-major
    # eigenvalues descending
    assert eigvals[0] >= eigvals[1] >= eigvals[2] >= eigvals[3]
    # rows orthonormal
    gram = top @ top.T
    assert torch.allclose(gram, torch.eye(2), atol=1e-5)


def test_load_subspaces_by_module_roundtrip(tmp_path):
    root = tmp_path / "repr_fim_subspace" / "M" / "refusal_train"
    root.mkdir(parents=True)
    saved = {
        "model.layers.0.self_attn.q_proj": torch.randn(3, 8),
        "model.layers.1.mlp.down_proj": torch.randn(3, 8),
    }
    torch.save(saved, root / "subspaces.pt")

    cfg = OmegaConf.create(
        {
            "model": {"short_name": "M"},
            "subspace_intervention": {"intervention_root": str(root), "top_k": 2},
        }
    )
    loaded = load_subspaces_by_module(cfg, "repr_fim_subspace")
    assert set(loaded.keys()) == {(0, "self_attn.q_proj"), (1, "mlp.down_proj")}
    # top_k=2 slices the leading rows
    assert loaded[(0, "self_attn.q_proj")].shape == (2, 8)

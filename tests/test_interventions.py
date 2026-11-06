"""Unit tests for subspace-intervention helpers, projection math, and hook wiring."""

import pytest
import torch
from omegaconf import OmegaConf

from interventions.base import (
    get_decoder_layers,
    parse_module_key,
    resolve_attr,
    slice_stored_basis,
)
from interventions.repr_fim import RepresentationFIMSubspaceIntervention


def _stub_model():
    block = torch.nn.Module()
    block.lin = torch.nn.Linear(4, 4, bias=False)
    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([block])
    model = torch.nn.Module()
    model.model = inner
    return model


_CFG = OmegaConf.create({"model": {"decoder_layers_attr": "model.layers"}})


def test_parse_module_key():
    assert parse_module_key("model.layers.5.self_attn.q_proj") == (5, "self_attn.q_proj")
    assert parse_module_key("base_model.model.model.layers.0.mlp.down_proj") == (0, "mlp.down_proj")


def test_parse_module_key_rejects_bad_path():
    with pytest.raises(ValueError):
        parse_module_key("model.embed_tokens")


def test_slice_stored_basis():
    basis = torch.arange(12).reshape(3, 4).float()
    assert torch.equal(slice_stored_basis(basis, None), basis)
    assert torch.equal(slice_stored_basis(basis, 2), basis[:2])
    with pytest.raises(ValueError):
        slice_stored_basis(basis, 5)


def test_get_decoder_layers_and_resolve_attr():
    model = _stub_model()
    layers = get_decoder_layers(model, _CFG)
    assert isinstance(layers, torch.nn.ModuleList) and len(layers) == 1
    assert isinstance(resolve_attr(layers[0], "lin"), torch.nn.Linear)


def test_repr_projection_apply_removes_in_subspace_component():
    # basis = first two standard axes; lambda=-1 should null those coordinates.
    basis = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    iv = RepresentationFIMSubspaceIntervention(
        lambda_scale=-1.0, subspaces_by_module={(0, "lin"): basis}
    )
    x = torch.ones(1, 4)
    out = iv.apply(x, layer_idx=0, target_module_name="lin")
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0, 1.0]]), atol=1e-6)


def test_repr_forward_pre_hook_shifts_module_input():
    model = _stub_model()
    lin = model.model.layers[0].lin
    with torch.no_grad():
        lin.weight.copy_(torch.eye(4))  # identity so output == (shifted) input
    basis = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    iv = RepresentationFIMSubspaceIntervention(
        lambda_scale=-1.0, subspaces_by_module={(0, "lin"): basis}
    )
    handles = iv.make_hook(_CFG, model)
    try:
        out = lin(torch.ones(1, 4))
    finally:
        for h in handles:
            h.remove()
    # identity weight => output equals the hook-shifted input.
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0, 1.0]]), atol=1e-6)


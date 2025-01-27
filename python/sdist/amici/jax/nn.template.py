# ruff: noqa: F401, F821, F841
import equinox as eqx
import jax.nn
import jax.random as jr
import jax
import amici.jax.nn


class TPL_MODEL_ID(eqx.Module):
    layers: dict
    inputs: list[str]
    outputs: list[str]

    def __init__(self, key):
        super().__init__()
        keys = jr.split(key, TPL_N_LAYERS)
        self.layers = {TPL_LAYERS}
        self.inputs = [TPL_INPUT]
        self.outputs = [TPL_OUTPUT]

    def forward(self, input, key=None):
        TPL_FORWARD
        return output


net = TPL_MODEL_ID

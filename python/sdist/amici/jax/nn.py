from pathlib import Path


import equinox as eqx
import jax.numpy as jnp

from amici._codegen.template import apply_template
from amici import amiciModulePath


class Flatten(eqx.Module):
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int, end_dim: int):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, x):
        if self.end_dim == -1:
            return jnp.reshape(x, x.shape[: self.start_dim] + (-1,))
        else:
            return jnp.reshape(
                x, x.shape[: self.start_dim] + (-1,) + x.shape[self.end_dim :]
            )


def tanhshrink(x: jnp.ndarray) -> jnp.ndarray:
    return x - jnp.tanh(x)


def generate_equinox(ml_model: "MLModel", filename: Path | str):  # noqa: F821
    # TODO: move to top level import and replace forward type definitions
    from petab_sciml import Layer

    filename = Path(filename)
    layer_indent = 12
    node_indent = 8

    layers = {layer.layer_id: layer for layer in ml_model.layers}

    tpl_data = {
        "MODEL_ID": ml_model.mlmodel_id,
        "LAYERS": ",\n".join(
            [
                _generate_layer(layer, layer_indent, ilayer)
                for ilayer, layer in enumerate(ml_model.layers)
            ]
        )[layer_indent:],
        "FORWARD": "\n".join(
            [
                _generate_forward(
                    node,
                    node_indent,
                    layers.get(
                        node.target,
                        Layer(layer_id="dummy", layer_type="Linear"),
                    ).layer_type,
                )
                for node in ml_model.forward
            ]
        )[node_indent:],
        "INPUT": ", ".join([f"'{inp.input_id}'" for inp in ml_model.inputs]),
        "OUTPUT": ", ".join(
            [
                f"'{arg}'"
                for arg in next(
                    node for node in ml_model.forward if node.op == "output"
                ).args
            ]
        ),
        "N_LAYERS": len(ml_model.layers),
    }

    filename.parent.mkdir(parents=True, exist_ok=True)

    apply_template(
        Path(amiciModulePath) / "jax" / "nn.template.py",
        filename,
        tpl_data,
    )


def _process_argval(v):
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, bool):
        return str(v)
    return str(v)


def _generate_layer(layer: "Layer", indent: int, ilayer: int) -> str:  # noqa: F821
    layer_map = {
        "Dropout1d": "eqx.nn.Dropout",
        "Dropout2d": "eqx.nn.Dropout",
        "Flatten": "amici.jax.nn.Flatten",
    }
    if layer.layer_type.startswith(
        ("BatchNorm", "AlphaDropout", "InstanceNorm")
    ):
        raise NotImplementedError(
            f"{layer.layer_type} layers currently not supported"
        )
    if layer.layer_type.startswith("MaxPool") and "dilation" in layer.args:
        raise NotImplementedError("MaxPool layers with dilation not supported")
    if layer.layer_type.startswith("Dropout") and "inplace" in layer.args:
        raise NotImplementedError("Dropout layers with inplace not supported")
    if layer.layer_type == "Bilinear":
        raise NotImplementedError("Bilinear layers not supported")

    kwarg_map = {
        "Linear": {
            "bias": "use_bias",
        },
        "Conv1d": {
            "bias": "use_bias",
        },
        "Conv2d": {
            "bias": "use_bias",
        },
        "LayerNorm": {
            "affine": "elementwise_affine",
            "normalized_shape": "shape",
        },
    }
    kwarg_ignore = {
        "Dropout1d": ("inplace",),
        "Dropout2d": ("inplace",),
    }
    kwargs = [
        f"{kwarg_map.get(layer.layer_type, {}).get(k, k)}={_process_argval(v)}"
        for k, v in layer.args.items()
        if k not in kwarg_ignore.get(layer.layer_type, ())
    ]
    # add key for initialization
    if layer.layer_type in (
        "Linear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
    ):
        kwargs += [f"key=keys[{ilayer}]"]
    type_str = layer_map.get(layer.layer_type, f"eqx.nn.{layer.layer_type}")
    layer_str = f"{type_str}({', '.join(kwargs)})"
    return f"{' ' * indent}'{layer.layer_id}': {layer_str}"


def _generate_forward(node: "Node", indent, layer_type=str) -> str:  # noqa: F821
    if node.op == "placeholder":
        # TODO: inconsistent target vs name
        return f"{' ' * indent}{node.name} = input"

    if node.op == "call_module":
        fun_str = f"self.layers['{node.target}']"
        if layer_type.startswith(("Conv", "Linear", "LayerNorm")):
            if layer_type in ("LayerNorm",):
                dims = f"len({fun_str}.shape)+1"
            if layer_type == "Linear":
                dims = 2
            if layer_type.endswith(("1d",)):
                dims = 3
            elif layer_type.endswith(("2d",)):
                dims = 4
            elif layer_type.endswith("3d"):
                dims = 5
            fun_str = f"(jax.vmap({fun_str}) if len({node.args[0]}.shape) == {dims} else {fun_str})"

    if node.op in ("call_function", "call_method"):
        map_fun = {
            "hardtanh": "jax.nn.hard_tanh",
            "hardsigmoid": "jax.nn.hard_sigmoid",
            "hardswish": "jax.nn.hard_swish",
            "tanhshrink": "amici.jax.nn.tanhshrink",
            "softsign": "jax.nn.soft_sign",
        }
        if node.target == "hardtanh":
            if node.kwargs.pop("min_val", -1.0) != -1.0:
                raise NotImplementedError(
                    "min_val != -1.0 not supported for hardtanh"
                )
            if node.kwargs.pop("max_val", 1.0) != 1.0:
                raise NotImplementedError(
                    "max_val != 1.0 not supported for hardtanh"
                )
        fun_str = map_fun.get(node.target, f"jax.nn.{node.target}")

    args = ", ".join([f"{arg}" for arg in node.args])
    kwargs = [
        f"{k}={v}" for k, v in node.kwargs.items() if k not in ("inplace",)
    ]
    if layer_type.startswith(("Dropout",)):
        kwargs += ["key=key"]
    kwargs_str = ", ".join(kwargs)
    if node.op in ("call_module", "call_function", "call_method"):
        return f"{' ' * indent}{node.name} = {fun_str}({args + ', ' + kwargs_str})"
    if node.op == "output":
        return f"{' ' * indent}{node.target} = {args}"

"""ONNX helpers for "xrock-model/1" onnx models (needs the `onnx` package).

A trained network is exported with its framework's exporter (e.g. torch.onnx.export) to the shape the C++ runner
expects: one float32 input [1, F] of raw features (normalization inside the graph), first output float32 [1, O].
`save_onnx_model_header` (model.py) then writes the JSON that carries the feature contract.
`linear_to_onnx` builds that shape for a linear model: the reference graph used to test the runner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def linear_to_onnx(path: str | Path, mean, std, weights, bias, opset: int = 17, window: int = 1,
                   features: int | None = None) -> None:
    """y = ((x - mean) / std) @ weights.T + bias as an ONNX graph with output "y" [1, O] and input "x" [1, F]
    (window 1) or [1, window, features] flattened oldest row first (the window contract of a sequence model)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    mean = np.asarray(mean, np.float32).reshape(1, -1)
    std = np.asarray(std, np.float32).reshape(1, -1)
    w = np.asarray(weights, np.float32)
    b = np.asarray(bias, np.float32).reshape(1, -1)
    f, o = mean.shape[1], w.shape[0]
    nodes, inits = [], []
    if window == 1:
        x_shape, flat = [1, f], "x"
    else:
        if features is None or window * features != f:
            raise ValueError("window * features must equal the parameter length")
        x_shape, flat = [1, window, features], "flat"
        nodes.append(helper.make_node("Reshape", ["x", "flat_shape"], ["flat"]))
        inits.append(numpy_helper.from_array(np.array([1, f], np.int64), "flat_shape"))
    graph = helper.make_graph(
        nodes + [
            helper.make_node("Sub", [flat, "mean"], ["centered"]),
            helper.make_node("Div", ["centered", "std"], ["z"]),
            helper.make_node("MatMul", ["z", "wt"], ["zw"]),
            helper.make_node("Add", ["zw", "b"], ["y"]),
        ],
        "xrock_linear",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, o])],
        inits + [numpy_helper.from_array(mean, "mean"), numpy_helper.from_array(std, "std"),
         numpy_helper.from_array(np.ascontiguousarray(w.T), "wt"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9  # readable by ONNX Runtime >= 1.16
    onnx.checker.check_model(model)
    onnx.save(model, str(path))

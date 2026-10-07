"""Python side of the xrock ML contract.

`xrock dataset` writes X.npy / Y.npy / ts.npy / meta.json ("xrock-dataset/1"); training reads them here and hands a
model back as an "xrock-model/1" JSON file, which `xrock model-eval` and the "model" strategy load. The feature list
and sampling grid travel inside both files, so a model can only be run on the features it was trained on.
"""

from .dataset import Dataset, load_dataset
from .model import save_linear_model, save_onnx_model_header

__all__ = ["Dataset", "load_dataset", "save_linear_model", "save_onnx_model_header"]
# ONNX graph helpers (need the onnx package): xrock_ml.onnx_export

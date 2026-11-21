"""Merge several SignSeek ONNX models into ONE model whose output is the mean of their probabilities.

    python model/ensemble.py OUT.onnx A.onnx B.onnx [C.onnx ...]

All models must take the same input `features [batch, L, F]` and output `probs [batch, C]`. Every node,
edge and initializer of model i is prefixed `m{i}_`, the shared input stays `features`, and
probs = (m0_probs + m1_probs + ...) / n. The browser loads it like any single model.
Needs only the `onnx` package.
"""

import sys

import onnx
from onnx import TensorProto, helper, numpy_helper

import numpy as np


def _dims(vi) -> list:
    return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in vi.type.tensor_type.shape.dim]


def merge(paths: list[str]) -> onnx.ModelProto:
    models = [onnx.load(p) for p in paths]
    inp = models[0].graph.input[0]
    for p, m in zip(paths, models):
        g = m.graph
        assert len(g.input) == 1 and g.input[0].name == "features", (p, [i.name for i in g.input])
        assert len(g.output) == 1 and g.output[0].name == "probs", (p, [o.name for o in g.output])
        assert _dims(g.input[0])[1:] == _dims(inp)[1:], f"{p}: input shape {_dims(g.input[0])} != {_dims(inp)}"
        assert _dims(g.output[0])[1:] == _dims(models[0].graph.output[0])[1:], f"{p}: output shape differs"

    nodes, inits, vinfo, outs = [], [], [], []
    for i, m in enumerate(models):
        g = m.graph
        pre = f"m{i}_"
        ren = lambda name: name if name in ("features", "") else pre + name  # noqa: E731
        for n in g.node:
            n2 = onnx.NodeProto()
            n2.CopyFrom(n)
            n2.name = pre + (n.name or n.op_type)
            n2.input[:] = [ren(x) for x in n.input]
            n2.output[:] = [ren(x) for x in n.output]
            nodes.append(n2)
        for t in g.initializer:
            t2 = onnx.TensorProto()
            t2.CopyFrom(t)
            t2.name = ren(t.name)
            inits.append(t2)
        for v in g.value_info:
            v2 = onnx.ValueInfoProto()
            v2.CopyFrom(v)
            v2.name = ren(v.name)
            vinfo.append(v2)
        outs.append(ren("probs"))

    # mean of the members' probabilities
    total = outs[0]
    for j, o in enumerate(outs[1:], 1):
        nodes.append(helper.make_node("Add", [total, o], [f"ens_sum{j}"], name=f"ens_add{j}"))
        total = f"ens_sum{j}"
    inits.append(numpy_helper.from_array(np.array(len(outs), dtype=np.float32), "ens_n"))
    nodes.append(helper.make_node("Div", [total, "ens_n"], ["probs"], name="ens_mean"))

    out_vi = onnx.ValueInfoProto()
    out_vi.CopyFrom(models[0].graph.output[0])
    graph = helper.make_graph(nodes, "signseek_ensemble", [inp], [out_vi], initializer=inits, value_info=vinfo)
    opset = max(o.version for o in models[0].opset_import if o.domain in ("", "ai.onnx"))
    merged = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                               producer_name="signseek-ensemble")
    merged.ir_version = models[0].ir_version
    merged.doc_string = "mean of: " + ", ".join(paths)
    onnx.checker.check_model(merged)
    assert merged.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    return merged


def main() -> None:
    out, *paths = sys.argv[1:]
    if len(paths) < 2:
        raise SystemExit("usage: ensemble.py OUT.onnx A.onnx B.onnx [...]")
    onnx.save(merge(paths), out)
    print(f"wrote {out}: mean of {len(paths)} models")


if __name__ == "__main__":
    main()

"""Per-layer HLS C++ emitters.

Every emitter mirrors the integer semantics in `models/executor.py` exactly:
int64 accumulation, fixed-point requantization with round-half-up, then clamp.
Pragmas come from the layer's Knobs. Nothing here is Vitis-only: the generated
code compiles with a plain C++ compiler, which ignores `#pragma HLS ...`.
"""
from __future__ import annotations

from ..ir import Add, Conv2d, Linear, MaxPool2d, ReLU


def qrange(bits: int, relu: bool = False) -> tuple[int, int]:
    return (0 if relu else -(1 << (bits - 1))), (1 << (bits - 1)) - 1


def format_array(name: str, values, per_line: int = 16) -> str:
    flat = [int(v) for v in values.reshape(-1)]
    lines, out = [], []
    for i in range(0, len(flat), per_line):
        lines.append("  " + ", ".join(str(v) for v in flat[i:i + per_line]))
    out.append(f"static const data_t {name}[{len(flat)}] = {{")
    out.append(",\n".join(lines))
    out.append("};")
    return "\n".join(out)


def _requant(node, indent: str, acc="acc", dst="v") -> str:
    """Emit the requantize + clamp sequence. `>>` is an arithmetic (floor) shift,
    matching torch.div(..., rounding_mode='floor') in the golden executor."""
    qmin, qmax = qrange(node.out_bits, node.relu)
    rnd = (1 << (node.shift - 1)) if node.shift > 0 else 0
    i = indent
    return "\n".join([
        f"{i}acc_t {dst} = ({acc} * {node.mult} + {rnd}) >> {node.shift};",
        f"{i}if ({dst} < {qmin}) {dst} = {qmin};",
        f"{i}if ({dst} > {qmax}) {dst} = {qmax};",
    ])


def _pipeline(knobs, indent: str) -> str:
    return f"{indent}#pragma HLS PIPELINE II=1" if knobs.pipeline else \
           f"{indent}#pragma HLS PIPELINE off"


def emit_conv(n: Conv2d) -> str:
    oh, ow = n.out_h(), n.out_w()
    in_elems = n.in_ch * n.in_h * n.in_w
    out_elems = n.out_ch * oh * ow
    w_name = f"{n.name}_w"
    u = max(1, n.knobs.unroll)
    body = [
        format_array(w_name, n.weight),
    ]
    if n.bias and n.bias_data is not None:
        body.append(format_array(f"{n.name}_b", n.bias_data))
    body.append(f"""
static void layer_{n.name}(const data_t in[{in_elems}], data_t out[{out_elems}]) {{
#pragma HLS ARRAY_PARTITION variable={w_name} cyclic factor={u} dim=1
  for (int oc = 0; oc < {n.out_ch}; ++oc) {{
    for (int oh = 0; oh < {oh}; ++oh) {{
      for (int ow = 0; ow < {ow}; ++ow) {{
{_pipeline(n.knobs, '        ')}
        acc_t acc = 0;
        for (int ic = 0; ic < {n.in_ch}; ++ic) {{
          for (int kh = 0; kh < {n.kh}; ++kh) {{
            for (int kw = 0; kw < {n.kw}; ++kw) {{
              #pragma HLS UNROLL factor={u}
              int ih = oh * {n.stride} - {n.pad} + kh;
              int iw = ow * {n.stride} - {n.pad} + kw;
              if (ih < 0 || ih >= {n.in_h} || iw < 0 || iw >= {n.in_w}) continue;
              acc_t w = (acc_t){w_name}[((oc * {n.in_ch} + ic) * {n.kh} + kh) * {n.kw} + kw];
              acc_t a = (acc_t)in[(ic * {n.in_h} + ih) * {n.in_w} + iw];
              acc += w * a;
            }}
          }}
        }}""")
    if n.bias and n.bias_data is not None:
        body.append(f"        acc += (acc_t){n.name}_b[oc];")
    body.append(_requant(n, "        "))
    body.append(f"""        out[(oc * {oh} + oh) * {ow} + ow] = (data_t)v;
      }}
    }}
  }}
}}""")
    return "\n".join(body)


def emit_linear(n: Linear) -> str:
    w_name = f"{n.name}_w"
    u = max(1, n.knobs.unroll)
    body = [format_array(w_name, n.weight)]
    if n.bias and n.bias_data is not None:
        body.append(format_array(f"{n.name}_b", n.bias_data))
    body.append(f"""
static void layer_{n.name}(const data_t in[{n.in_features}], data_t out[{n.out_features}]) {{
#pragma HLS ARRAY_PARTITION variable={w_name} cyclic factor={u} dim=1
  for (int oc = 0; oc < {n.out_features}; ++oc) {{
{_pipeline(n.knobs, '    ')}
    acc_t acc = 0;
    for (int i = 0; i < {n.in_features}; ++i) {{
      #pragma HLS UNROLL factor={u}
      acc += (acc_t){w_name}[oc * {n.in_features} + i] * (acc_t)in[i];
    }}""")
    if n.bias and n.bias_data is not None:
        body.append(f"    acc += (acc_t){n.name}_b[oc];")
    body.append(_requant(n, "    "))
    body.append("""    out[oc] = (data_t)v;
  }
}""")
    return "\n".join(body)


def emit_relu(n: ReLU) -> str:
    qmin, qmax = qrange(n.bits, relu=True)
    u = max(1, n.knobs.unroll)
    return f"""
static void layer_{n.name}(const data_t in[{n.numel}], data_t out[{n.numel}]) {{
  for (int i = 0; i < {n.numel}; ++i) {{
{_pipeline(n.knobs, '    ')}
    #pragma HLS UNROLL factor={u}
    acc_t v = (acc_t)in[i];
    if (v < {qmin}) v = {qmin};
    if (v > {qmax}) v = {qmax};
    out[i] = (data_t)v;
  }}
}}"""


def emit_maxpool(n: MaxPool2d) -> str:
    oh, ow = n.out_h(), n.out_w()
    in_elems = n.in_ch * n.in_h * n.in_w
    out_elems = n.in_ch * oh * ow
    u = max(1, n.knobs.unroll)
    return f"""
static void layer_{n.name}(const data_t in[{in_elems}], data_t out[{out_elems}]) {{
  for (int c = 0; c < {n.in_ch}; ++c) {{
    for (int oh = 0; oh < {oh}; ++oh) {{
      for (int ow = 0; ow < {ow}; ++ow) {{
{_pipeline(n.knobs, '        ')}
        data_t best = in[(c * {n.in_h} + oh * {n.stride}) * {n.in_w} + ow * {n.stride}];
        for (int kh = 0; kh < {n.kh}; ++kh) {{
          for (int kw = 0; kw < {n.kw}; ++kw) {{
            #pragma HLS UNROLL factor={u}
            data_t cand = in[(c * {n.in_h} + oh * {n.stride} + kh) * {n.in_w}
                             + ow * {n.stride} + kw];
            if (cand > best) best = cand;
          }}
        }}
        out[(c * {oh} + oh) * {ow} + ow] = best;
      }}
    }}
  }}
}}"""


def emit_add(n: Add) -> str:
    qmin, qmax = qrange(n.bits)
    u = max(1, n.knobs.unroll)
    return f"""
static void layer_{n.name}(const data_t a[{n.numel}], const data_t b[{n.numel}],
                           data_t out[{n.numel}]) {{
  for (int i = 0; i < {n.numel}; ++i) {{
{_pipeline(n.knobs, '    ')}
    #pragma HLS UNROLL factor={u}
    acc_t v = (acc_t)a[i] + (acc_t)b[i];
    if (v < {qmin}) v = {qmin};
    if (v > {qmax}) v = {qmax};
    out[i] = (data_t)v;
  }}
}}"""


EMITTERS = {
    Conv2d: emit_conv,
    Linear: emit_linear,
    ReLU: emit_relu,
    MaxPool2d: emit_maxpool,
    Add: emit_add,
}


def emit_layer(node) -> str:
    try:
        return EMITTERS[type(node)](node)
    except KeyError:
        raise TypeError(f"no emitter for {type(node).__name__}")

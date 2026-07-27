"""Target device fabric budgets.

Numbers are published fabric totals for the programmable logic (not counting the
ARM PS). Sources:
- Zynq-7020 (XC7Z020): Xilinx DS190 Zynq-7000 datasheet — 53200 LUT, 106400 FF,
  220 DSP48E1, 140 BRAM36 (== 280 BRAM18).
- Ultra96 / ZU3EG (XCZU3EG): Zynq UltraScale+ DS891 — 70560 LUT, 141120 FF,
  360 DSP48E2, 216 BRAM36 (== 432 BRAM18).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceBudget:
    name: str
    lut: int
    ff: int
    dsp: int
    bram18: int


DEVICES: dict[str, DeviceBudget] = {
    "zynq-7020": DeviceBudget("zynq-7020", lut=53200, ff=106400, dsp=220, bram18=280),
    "ultra96": DeviceBudget("ultra96", lut=70560, ff=141120, dsp=360, bram18=432),
    "pynq-z2": DeviceBudget("pynq-z2", lut=53200, ff=106400, dsp=220, bram18=280),
}


def get_device(name: str) -> DeviceBudget:
    try:
        return DEVICES[name]
    except KeyError:
        raise ValueError(f"unknown device {name!r}; known: {sorted(DEVICES)}")

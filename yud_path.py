"""Location of the packed York Urban annotations (made by pack_yud.py).
Set the environment variable YUD_PACKED, or put the file at data/yud_packed.npz."""
import os
YUD = os.environ.get("YUD_PACKED",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "yud_packed.npz"))
if not os.path.exists(YUD):
    raise SystemExit("York Urban annotations not found at " + YUD +
                     "\nSee README.md, section 'The York Urban Database', for how to make this file.")

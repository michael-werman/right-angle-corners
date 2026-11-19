"""Shared figure style: TrueType (Type 42) fonts with a proper Unicode mapping (STIX), so that the text of every
figure can be searched and copied."""
import matplotlib as mpl
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "axes.formatter.use_mathtext": True, "axes.unicode_minus": True})

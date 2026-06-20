"""CIF observation helper.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file converts structures to CIF text for LLM-agent workflows.
"""

from io import StringIO

from ase.io import write


def atoms_to_cif(atoms) -> str:
    """Serialize ASE atoms to a CIF string."""
    buffer = StringIO()
    write(buffer, atoms, format="cif")
    return buffer.getvalue()

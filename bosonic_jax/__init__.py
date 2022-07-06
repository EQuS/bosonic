"""
Bosonic Codes
"""
import os
from .codes import *
from .utils import *
from .simulator import *
from .jax_qutip import *

with open(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "VERSION.txt")), "r"
) as _ver_file:
    __version__ = _ver_file.read().rstrip()

__author__ = "Shantanu Jha"
__credits__ = "Bosonic Codes Dev Team"

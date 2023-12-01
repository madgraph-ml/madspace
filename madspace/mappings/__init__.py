""" Mappings for the MadSpace """

# Load the subfolders
from . import functional

# Import the base class first
from .base import *

# Then all inheriting modules
from .phasespace import *
from .helper import *
from .propagators import *
from .twoparticle import *

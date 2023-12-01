r"""
=============================================
                  _ __                      
  /\/\   __ _  __| / _\_ __   __ _  ___ ___ 
 /    \ / _` |/ _` \ \| '_ \ / _` |/ __/ _ \
/ /\/\ \ (_| | (_| |\ \ |_) | (_| | (_|  __/
\/    \/\__,_|\__,_\__/ .__/ \__,_|\___\___|
                      |_|                                  
=============================================

Modules to construct differentiable and GPU-ready 
phase-space mappings susing PyTorch.

"""
from . import mappings

__all__ = ["distributions", "mappings", "models", "plotting", "training"]

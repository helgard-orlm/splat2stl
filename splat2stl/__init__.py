"""splat2stl: 3D Gaussian Splat (.ply) -> watertight, printable STL/OBJ via silhouette carving."""
from .hull import splat_to_mesh

__all__ = ["splat_to_mesh"]
__version__ = "0.1.0"

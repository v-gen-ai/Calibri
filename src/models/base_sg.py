# src/models/base_pipeline.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class BaseSGPipeline(ABC):
    @abstractmethod
    def flatten_coefficients(self) -> np.ndarray:
        """Return current coefficients as flat numpy vector"""
        pass  

    @abstractmethod
    def apply_coefficients(self, x: np.ndarray) -> None:
        """Apply flat vector to the model (renamed from unflatten_coefficients)"""
        pass  

    @abstractmethod
    def flat_to_struct(self, x: Optional[np.ndarray] = None) -> List[Dict[str, List]]:
        """
        Return structured representation (list per model) as Dict[str, List], e.g.:
        - FLUX: {"double": List[[attn, mlp]], "single": List[float], "model": [float]}
        - SD3:  {"main": List[[attn, mlp]], "context": List[[attn, mlp]], "attn2": List[float], "model": [float]}
        If x is None, use current coefficients.  
        """
        pass  

    @abstractmethod
    def struct_to_flat(self, s: List[Dict[str, List]]) -> np.ndarray:
        """Pack structured representation back to a flat vector"""  
        pass  

    @abstractmethod
    def get_coefficient_shapes(self) -> Dict[str, int]:
        """Return dimensions metadata used by pack/unpack helpers"""  
        pass  

    @abstractmethod
    def __call__(self, prompts, num_inference_steps, guidance_scale, height, width, generator, **kwargs):
        pass

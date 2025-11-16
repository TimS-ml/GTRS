# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
SwiGLU feed-forward network.
"""

import os
from typing import Callable, Optional
import warnings

from torch import Tensor, nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    Swi G L U F F N.
    """
    def __init__(
        self,
        in_features: int,
        """
        Initialize the instance.
        
        Args:
            in_features: In features.
            hidden_features: Hidden features.
            out_features: Out features.
            act_layer: Act layer.
            drop: Drop.
            bias: Bias.
        """
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import SwiGLU

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (SwiGLU)")
    else:
        """
        Initialize the instance.
        
        Args:
            in_features: In features.
            hidden_features: Hidden features.
            out_features: Out features.
            act_layer: Act layer.
            drop: Drop.
            bias: Bias.
        """
        warnings.warn("xFormers is disabled (SwiGLU)")
        raise ImportError
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (SwiGLU)")


class SwiGLUFFNFused(SwiGLU):
    """
    Swi G L U F F N Fused.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )

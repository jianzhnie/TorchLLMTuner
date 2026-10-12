"""Expert parallelism: swap HF MoE blocks for the EP-capable llmtuner MoE.

``expert_parallel.py`` is the public entry. ``probe.py`` inspects HF layouts,
``convert.py`` builds native MoE blocks, ``swap.py`` orchestrates replacement,
and ``ckpt.py`` owns expert checkpoint conversion.
"""

from .expert_parallel import *  # noqa: F401,F403
from .expert_parallel import __all__ as __all__

"""Expert parallelism: swap HF MoE blocks for the EP-capable llmtuner MoE.

``expert_parallel.py`` is the public entry. ``hf_inspection.py`` inspects HF
layouts, ``block_conversion.py`` constructs native MoE blocks,
``model_replacement.py`` replaces them in a complete model, and
``checkpointing.py`` converts expert checkpoint state.
"""

from .expert_parallel import *  # noqa: F401,F403
from .expert_parallel import __all__ as __all__

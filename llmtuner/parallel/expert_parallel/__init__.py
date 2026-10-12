"""Expert parallelism: swap HF MoE blocks for the EP-capable llmtuner MoE.

EP core idea (MoE): shard experts across ranks; an all-to-all routes each token
to its expert's rank and back. ``swap.py`` is the weight-moving swap itself;
``apply.py`` wires it onto a model given the EP process group.

The two entry points are re-exported lazily (PEP 562): ``swap.py`` pulls in the
whole MoE stack (``models/common/moe`` and its custom ops), and lighter
submodules like ``ckpt.py`` -- imported by the checkpointer and optimizer
containers -- must stay loadable on hosts whose torch predates those ops.
"""

from .expert_parallel import *  # noqa: F401,F403
from .expert_parallel import __all__ as __all__

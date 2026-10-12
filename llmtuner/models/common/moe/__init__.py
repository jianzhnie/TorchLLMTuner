"""The MoE stack, one node per file.

Split out of torchtitan's single ``models/common/moe.py`` (plus its
``token_dispatcher.py``), which carried all of this in one module:

* ``routing.py`` -- ``TokenChoiceTopKRouter``, ``QuantileBalancedTopKRouter``,
  ``QuantileBalancer``. Token-choice routing, including node-limited groups.
* ``experts.py`` -- ``GroupedExperts`` (every expert's weights as three stacked
  tensors) and ``RoutedExperts`` (the dispatch/combine pair around them).
* ``layer.py`` -- ``MoE`` itself, plus ``iter_moe_layers`` /
  ``MOE_LAYER_ATTRS``, the walk the hooks and the compile pass use.
* ``auxiliary_loss.py`` -- ``MicrobatchWiseLoadBalanceLoss``, the per-forward
  sequence-wise auxiliary loss a router can carry.
* ``bias_update.py`` -- the auxiliary-loss-free bias updates and optimizer hooks.
  the optimizer step pre-hooks that call it, sign-based and quantile.

Local and distributed token dispatch implementations and their metadata live
in ``parallel/expert_parallel/token_dispatch.py``. This package re-exports
nothing; import the defining module directly. Model APIs remain discoverable
through the ``models.common`` index.
"""

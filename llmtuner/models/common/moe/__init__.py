"""The MoE stack, one node per file.

Split out of torchtitan's single ``models/common/moe.py`` (plus its
``token_dispatcher.py``), which carried all of this in one module:

* ``routing.py`` -- ``TokenChoiceTopKRouter``, ``QuantileBalancedTopKRouter``,
  ``QuantileBalancer``. Token-choice routing, including node-limited groups.
* ``experts.py`` -- ``GroupedExperts`` (every expert's weights as three stacked
  tensors) and ``RoutedExperts`` (the dispatch/combine pair around them).
* ``token_dispatch.py`` -- local and distributed token dispatch implementations.
  ``TorchAOTokenDispatcher`` and the metadata they hand to ``combine``.
* ``layer.py`` -- ``MoE`` itself, plus ``iter_moe_layers`` /
  ``MOE_LAYER_ATTRS``, the walk the hooks and the compile pass use.
* ``auxiliary_loss.py`` -- ``MicrobatchWiseLoadBalanceLoss``, the per-forward
  sequence-wise auxiliary loss a router can carry.
* ``bias_update.py`` -- the auxiliary-loss-free bias updates and optimizer hooks.
  the optimizer step pre-hooks that call it, sign-based and quantile.

Like the other indexes in this package, this one re-exports nothing: importing
one node must not drag in its siblings (``dispatcher`` pulls in the EP
collectives, ``load_balance`` the SPMD mesh context), so import the leaf you
want. The flat names stay reachable from the ``models.common`` index, which is
the package's discoverability surface.
"""

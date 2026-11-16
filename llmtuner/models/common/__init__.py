"""Shared components used by explicit model implementations.

``activation`` and ``linear`` define shared projection mathematics;
``attention``, ``input_preparation`` and ``targets`` preserve token semantics.
``aux_loss`` and ``gradient_norm`` define loss and logical gradient behavior.
The ``moe`` package owns routing, expert computation and balancing contracts.

Import from defining leaf modules. This package does not re-export components
or load optional model and accelerator dependencies.
"""

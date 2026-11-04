"""Model registration, HF adaptation, and reusable model components.

Responsibilities are split across three layers:

* ``architecture.py`` -- the lightweight registration seam for custom HF types.
* ``hf/`` -- the adaptation layer: ``factory.py`` builds the HF config and
  resolves the model class, ``model.py`` holds the ``HFTransformerModel`` the
  trainer drives, ``flops.py`` counts model FLOPs per token for the MFU
  numerator, and
  ``state_dict_adapter.py`` maps the checkpoint FQNs. Runtime Hugging Face
  adaptation belongs here.
* ``common/`` -- the model vocabulary, vendored from torchtitan
  ``models/common/``: attention pieces (``attention/``), MoE (``moe/``),
  feed-forward, linear, embedding, RoPE, activations, the aux-loss carrier and
  the multimodal glue. Nothing here knows which family a model belongs to.

Two conventions, both inherited from ``common/``: a package index is for
discoverability while a leaf module is what you import, and a component with
real behavior of its own gets its own file.
"""

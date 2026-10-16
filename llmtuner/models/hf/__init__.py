"""The Hugging Face adaptation layer: build it, run it, count it, checkpoint it.

Main modules:

* ``factory.py`` -- builds the HF ``PretrainedConfig`` (offline architecture,
  hub id or local checkpoint) and resolves the ``ForCausalLM`` class.
* ``materialization.py`` -- allocates meta model storage after parallelism,
  rebuilds runtime buffers, and restores shared parameter and buffer aliases.
* ``model_setup.py`` -- validates attention heads and experts kernels, resolves
  decoder submodule names, and applies the optional lm-head dtype cast.
* ``attention_backend.py`` -- selects and registers the HF attention adapter.
* ``model.py`` -- ``HFTransformerModel``: the five-part contract
  (``tok_embeddings`` / ``layers`` / ``norm`` / ``lm_head`` / ``rotary_emb``),
  the packed-document attention wiring, and the forward the trainer calls.
  This is the consumer of everything else here, and its name follows upstream's
  ``experiments/transformers_modeling_backend/model.py``.
* ``flops.py`` -- training FLOPs per token, the MFU denominator. Pure
  arithmetic over an HF config, kept apart from the building; the trainer's
  runtime assembly passes the already resolved text config directly to
  ``flops_per_token``.
* ``state_dict_adapter.py`` -- ``HFTransformerStateDictAdapter``: the model's
  state-dict FQNs to standard HF safetensors keys.

Like the other indexes in this package, this one re-exports nothing: importing
one node must not drag in its siblings (``model`` pulls in the whole Hugging
Face modeling stack, ``factory``/``flops`` need only the config classes), so
import the leaf you want.
"""

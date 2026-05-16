# CLAUDE.md

IF3270 Tugas Besar 2 — CNN (Intel Image Classification) + Image Captioning (RNN/LSTM, Flickr8k). Keras training + a from-scratch NumPy framework (`Diffable` DAG) for forward/backward.

## Layout

```
src/
  algorithm/
    core/          diffable.py, model.py, criterion.py, optimizer.py, parameter.py
    function/      relu, sigmoid, tanh, softmax
    neural/        linear, conv, local, maxpool, avgpool, reshape, pad, permute,
                   concat, select, embedding, rnn, lstm
    criterion/     mse, mae, rmse, bce, cce, scce, cos
    optimizer/     sgd, adam, rmsprop
    utility/       image_utils
  compat/          LocallyConnected2D.py  ← custom Keras 3 LC2D (valid, stride=1)
  notebook/
    Tubes2.ipynb
    training/      cnn_training.py, captioning_preprocessing.py, captioning_training.py
    evaluation/    cnn_evaluation.py, captioning_evaluation.py
dataset/           download_datasets.py  (gitignored data: intel-image-classification/, flickr8k/)
models/            cnn_*.keras + training_results.json (gitignored)
models_captioning/ cap_*.keras + captioning_results.json (gitignored)
data_processed/    captioning preprocessed npy/pkl/json (gitignored)
```

## Diffable framework

Static computation graph: each node has `_sources` (inputs) and `_sinks` (outputs).
- `_calculate_value(sources)` → forward
- `_calculate_gradient(sources, value)` → backward (returns `{source_node: grad}`)
- `_state` dict for caching intermediates (use this, not new self.attrs — `__setattr__` warns)
- Leaves: `Parameter` (persists weights), `Input` (data assigned via `node._value = x`)
- `Model.train_step` builds criterion → forward → backward via `get_gradient(param)` → `optimizer.step`

## CNN

**Keras training** (`cnn_training.py`): 16 variants over `n_blocks × filters × kernel_size × pool_type`. Conv2D uses `padding='same'`. After training, also trains a Locally-Connected version of the best model.

**Locally-connected training** uses `src/compat/LocallyConnected2D.py` (custom Keras 3 layer, **always valid + stride 1**) preceded by `layers.ZeroPadding2D(padding=((kH-1)//2, (kW-1)//2))` to emulate 'same'. So the trained model's layer sequence is `... → ZeroPadding2D → LocallyConnected2D → Pool → ...`.

**From-scratch eval** (`cnn_evaluation.py`):
- `build_scratch_model(keras_model)` walks `keras_model.layers` and emits Diffable nodes for each layer type. Must handle: `Conv2D`, our custom `LocallyConnected2D`, `MaxPooling2D`, `AveragePooling2D`, `Flatten`, `Dense`, `Dropout` (skip), `InputLayer` (skip), `ZeroPadding2D`.
- For `LocallyConnected2D`: Keras kernel is `(out_H, out_W, kH*kW*C_in, C_out)`; our `LocallyConnected2D` node expects flat `(out_H*out_W, klen, C_out)`. Built with `padding='valid'`, `strides=(1,1)` because that's what the custom Keras layer does — surrounding padding comes from a separate `ZeroPadding2D` layer.

**LC2D forward** (`src/algorithm/neural/local.py`): uses `_im2col` + `einsum('npi,pio->npo', col, kernel)`. The number of positions `n_pos` in the kernel must match `out_H*out_W` from the im2col output. If they mismatch, padding handling is wrong somewhere upstream.

## Captioning

Two inject modes (`captioning_training.py::build_model(..., inject_mode=...)`):
- **`'pre'`** — image feature → `Dense(embed)` → expand to `(B,1,E)` → concat with caption embedding → stacked RNN/LSTM (return sequences) → `Lambda` slice `[1:max_len+1]` → `Dense(vocab)`.
- **`'init'`** — image feature → `Dense(units, tanh, name='h_init')` (and `'c_init'` for LSTM) → used as `initial_state` of first recurrent layer only (deeper layers start at zero); caption embedding fed directly, no concat, no slice. `mask_zero=False` here because Keras forbids combining a masked embedding with `initial_state`.

Tag format: `{inject_mode}_{cell_type}_l{n_layers}_u{units}` (e.g. `init_lstm_l2_u512`). Saved model file is `cap_{tag}.keras`. Results JSON has `inject_mode` field (defaults to `'pre'` when missing for backward-compat).

`ScratchDecoder` (in `captioning_evaluation.py`) supports both inject modes via the `inject_mode` constructor arg, plus two decoding strategies:
- `generate(img_feat, max_len)` — greedy argmax.
- `generate_beam(img_feat, max_len, beam_size, length_norm=0.7)` — beam search with length-normalised log-prob scoring; tracks per-beam hidden/cell states.

**Critical: slice offset.** With `dec_input = [<start>, w₀, w₁, ...]` and `dec_target = [w₀, w₁, ..., <end>]`, the concatenated sequence is `[img, <start>, w₀, ...]` (length `max_len+1`). The slice must be `[1:max_len+1]` so that `output[t] = h[t+1] = f(img, <start>, w₀..w_{t-1})` predicts `dec_target[t] = w_t`. Slicing `[:max_len]` is off-by-one — it forces `h[1]=f(img,<start>)` to predict `w_1` when it just saw `<start>`, which destroys learning. The from-scratch `ScratchDecoder` must mirror this: feed `img_proj`, then `embed(<start>)`, then iterate predict-and-feed. The Diffable `_SliceSeq` takes a `start=1` argument for the same reason.

**RNN/LSTM nodes** (`rnn.py`, `lstm.py`):
- Unrolled in-place inside `_calculate_value` — graph is static, sequence loop is internal.
- Save BPTT cache into `self._state` only.
- LSTM gate order matches Keras: `[i, f, g/c, o]`. Bias may be shape `(2, 4u)` (Keras `implementation=2`); summed across rows in forward, restored in backward.
- Sources: `(x_seq, W_x, W_h, b)` for RNN; `(x_seq, W, U, b)` for LSTM. Weight ordering matches `keras_layer.get_weights()`.

**From-scratch eval** (`captioning_evaluation.py`):
- `ScratchDecoder`: raw NumPy greedy step-by-step decoder (no Diffable graph) — fast inference.
- `build_diffable_decoder`: full Diffable graph including helper nodes `_ExpandDim`, `_SliceSeq` (`_warn_on_unmanaged_state = False` because it stores `_len` before `super().__init__()`), `_SeqLinear`.

## Path conventions

Training and evaluation scripts live at `src/notebook/training|evaluation/`, so:
```python
REPO_ROOT = Path(__file__).resolve().parents[3]
```
Datasets at `dataset/intel-image-classification/` and `dataset/flickr8k/`.

## Datasets

Run `python dataset/download_datasets.py` (uses `kagglehub`, requires `~/.kaggle/kaggle.json`). Both `dataset/intel-image-classification/` and `dataset/flickr8k/` are gitignored.

## Style notes

Code in this repo intentionally has minimal comments — no section dividers (`# ─── X ───`), no inline shape annotations on trivial lines, no step-by-step gradient math comments. Keep it that way.

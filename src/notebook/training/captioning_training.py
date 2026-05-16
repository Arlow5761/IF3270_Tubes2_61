"""
Image Captioning — Keras Training: 12 Architecture Variants (IF3270 Tubes 2)
=============================================================================
Trains 12 decoder variants (RNN + LSTM × 3 num_layers × 2 hidden_units).

Prerequisites:
    python src/notebook/captioning_preprocessing.py   (run first)

Run:
    python src/notebook/captioning_training.py
"""

import json
import time
import itertools
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

REPO_ROOT  = Path(__file__).resolve().parents[3]
PROC_DIR   = REPO_ROOT / 'data_processed'
MODELS_DIR = REPO_ROOT / 'models_captioning'
MODELS_DIR.mkdir(exist_ok=True)

EMBED_DIM   = 256
EPOCHS      = 30
BATCH_SIZE  = 64
LEARNING_RATE = 1e-3

# Experiment grid: 2 inject modes × 2 cell × 3 layers × 2 hidden = 24
# 'pre'  = pre-inject (image concatenated as first timestep)
# 'init' = init-inject (image projected to RNN initial hidden state)
INJECT_MODES = ['pre', 'init']
CELL_TYPES   = ['rnn', 'lstm']
N_LAYERS     = [1, 2, 3]
HIDDEN_DIMS  = [128, 512]


def load_data(split: str):
    img_f   = np.load(PROC_DIR / f'{split}_img_feats.npy')
    dec_in  = np.load(PROC_DIR / f'{split}_dec_input.npy')
    dec_tgt = np.load(PROC_DIR / f'{split}_dec_target.npy')
    return img_f, dec_in, dec_tgt



def build_model(vocab_size: int, feature_dim: int, embed_dim: int,
                units: int, n_layers: int, cell_type: str,
                max_len: int, inject_mode: str = 'pre') -> keras.Model:
    """Encoder-decoder captioning model.

    inject_mode:
      'pre'  — image projected to (B,1,E) and concatenated as first timestep;
               output sliced [1:max_len+1] so each h[t+1] predicts target[t].
      'init' — image projected to (B,units) as initial hidden (and cell, for
               LSTM) state of the first recurrent layer; deeper layers start
               at zero. Output aligned directly with target — no slice needed.

    Inputs:
        img_feat  : (batch, feature_dim)
        dec_input : (batch, max_len)  [<start>, w₀, ..., w_{max_len-2}]
    Output:
        predictions: (batch, max_len, vocab_size)
    """
    img_feat  = keras.Input(shape=(feature_dim,), name='img_feat')
    dec_input = keras.Input(shape=(max_len,), name='dec_input')

    if inject_mode == 'pre':
        img_proj = layers.Dense(embed_dim, activation='relu', name='img_proj')(img_feat)
        img_proj = layers.Reshape((1, embed_dim))(img_proj)
        cap_emb  = layers.Embedding(vocab_size, embed_dim,
                                    mask_zero=True, name='embedding')(dec_input)
        x = layers.Concatenate(axis=1)([img_proj, cap_emb])

        for i in range(n_layers):
            if cell_type == 'lstm':
                x = layers.LSTM(units, return_sequences=True, name=f'lstm_{i}')(x)
            else:
                x = layers.SimpleRNN(units, return_sequences=True,
                                     activation='tanh', name=f'rnn_{i}')(x)

        x   = layers.Lambda(lambda t: t[:, 1:max_len + 1, :], name='slice')(x)
        out = layers.Dense(vocab_size, activation='softmax', name='output')(x)

    elif inject_mode == 'init':
        # Image → initial state(s) for first recurrent layer.
        h_init = layers.Dense(units, activation='tanh', name='h_init')(img_feat)
        if cell_type == 'lstm':
            x = layers.LSTM(units, return_sequences=True,
                            name=f'lstm_{i}')(x)
        else:
            x = layers.SimpleRNN(units, return_sequences=True,
                                 activation='tanh', name=f'rnn_{i}')(x)

    # drop the image timestep; output[t] = state after seeing img + dec_input[0..t]
    # so output[t] predicts dec_target[t] given the correct history
    x   = layers.Lambda(lambda t: t[:, 1:max_len + 1, :], name='slice')(x)
    out = layers.Dense(vocab_size, activation='softmax', name='output')(x)

    return keras.Model(inputs=[img_feat, dec_input], outputs=out,
                       name=f'cap_{inject_mode}_{cell_type}_l{n_layers}_u{units}')



def greedy_decode_batch(model: keras.Model, img_feats: np.ndarray,
                        vocab: dict, id2word: dict, max_len: int,
                        batch_size: int = 128) -> list:
    """Greedy decode: produce one caption per image."""
    pad_id   = vocab['<pad>']
    start_id = vocab['<start>']
    end_id   = vocab['<end>']
    N = len(img_feats)
    all_captions = []

    for start in range(0, N, batch_size):
        feats = img_feats[start:start + batch_size]
        B = len(feats)
        tokens = np.full((B, max_len), pad_id, dtype=np.int32)
        tokens[:, 0] = start_id
        finished = np.zeros(B, dtype=bool)

        for t in range(max_len - 1):
            preds    = model.predict([feats, tokens], verbose=0)
            next_tok = preds[:, t, :].argmax(axis=-1)
            for b in range(B):
                if not finished[b]:
                    tokens[b, t + 1] = next_tok[b]
                    if next_tok[b] == end_id:
                        finished[b] = True
            if finished.all():
                break

        for b in range(B):
            words = []
            for t in range(1, max_len):
                tok = tokens[b, t]
                if tok in (end_id, pad_id):
                    break
                words.append(id2word.get(int(tok), '<unk>'))
            all_captions.append(words)

    return all_captions



def compute_bleu4(hypotheses: list, references: list) -> float:
    """Compute corpus BLEU-4.  references: list of list-of-lists (multiple refs/image)."""
    smooth = SmoothingFunction().method1
    return corpus_bleu(references, hypotheses,
                       weights=(0.25, 0.25, 0.25, 0.25),
                       smoothing_function=smooth)


def main():
    print(f'TensorFlow {tf.__version__}')

    train_img, train_in, train_tgt = load_data('train')
    val_img,   val_in,   val_tgt   = load_data('val')

    feature_dim = train_img.shape[1]
    max_len     = train_in.shape[1]

    with open(PROC_DIR / 'vocab.json') as f:
        vocab = json.load(f)
    id2word = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    pad_id     = vocab['<pad>']

    print(f'vocab_size={vocab_size}  feature_dim={feature_dim}  max_len={max_len}')

    def make_weights(targets):
        return (targets != pad_id).astype(np.float32)

    train_w = make_weights(train_tgt)
    val_w   = make_weights(val_tgt)

    # Load all unique image features from preprocessing step
    with open(PROC_DIR / 'image_features.pkl', 'rb') as f:
        all_image_features = pickle.load(f)

    # Load test references to get unique test image names and their captions
    ref_path = PROC_DIR / 'test_references.json'
    with open(ref_path) as f:
        ref_map = json.load(f)

    # Get the unique test image names in a sorted order from references
    unique_test_image_names_from_refs = sorted(ref_map.keys())

    # Prepare image features for greedy decoding for these unique test images
    # Filter to ensure only images with extracted features are considered for BLEU calculation.
    test_img_for_bleu = []
    filtered_unique_test_image_names = []
    for img_name in unique_test_image_names_from_refs:
        if img_name in all_image_features: # Check if feature exists for this image
            test_img_for_bleu.append(all_image_features[img_name])
            filtered_unique_test_image_names.append(img_name)
    test_img_for_bleu = np.array(test_img_for_bleu)

    print(f'train={len(train_img)}  val={len(val_img)}  test_unique_images_for_bleu={len(test_img_for_bleu)}')

    results = []

    for inject_mode, cell_type, n_layers, units in itertools.product(
            INJECT_MODES, CELL_TYPES, N_LAYERS, HIDDEN_DIMS):
        tag = f'{inject_mode}_{cell_type}_l{n_layers}_u{units}'
        print(f'\n{"─"*60}')
        print(f'[{tag}]')

        model = build_model(vocab_size, feature_dim, EMBED_DIM,
                            units, n_layers, cell_type, max_len,
                            inject_mode=inject_mode)
        model.compile(
            optimizer = keras.optimizers.Adam(LEARNING_RATE),
            loss      = keras.losses.SparseCategoricalCrossentropy(),
        )
        model.summary(print_fn=lambda s: None)

        t0 = time.time()
        history = model.fit(
            x             = [train_img, train_in],
            y             = train_tgt,
            sample_weight = train_w,
            validation_data = ([val_img, val_in], val_tgt, val_w),
            epochs        = EPOCHS,
            batch_size    = BATCH_SIZE,
            callbacks     = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            ],
            verbose=1,
        )
        train_time = time.time() - t0

        # Generate hypotheses for the unique test images
        hyps  = greedy_decode_batch(model, test_img_for_bleu, vocab, id2word, max_len)

        # Prepare references for these same unique test images, ensuring order matches
        refs_aligned = [[r.split() for r in ref_map.get(img, [''])]
                        for img in filtered_unique_test_image_names]

        bleu4 = compute_bleu4(hyps, refs_aligned)

        print(f'  test BLEU-4 = {bleu4:.4f}   train_time = {train_time:.0f}s')

        save_path = MODELS_DIR / f'cap_{tag}.keras'
        model.save(str(save_path))

        results.append({
            'tag':         tag,
            'inject_mode': inject_mode,
            'cell_type':   cell_type,
            'n_layers':    n_layers,
            'units':       units,
            'train_loss':  history.history['loss'],
            'val_loss':    history.history['val_loss'],
            'test_bleu4':  float(bleu4),
            'train_time':  float(train_time),
            'saved_to':    str(save_path.relative_to(MODELS_DIR)),
            'n_params':    model.count_params(),
        })

    summary_path = MODELS_DIR / 'captioning_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAll results saved → {summary_path}')

    results_sorted = sorted(results, key=lambda r: r['test_bleu4'], reverse=True)
    print('\n── Ranking by BLEU-4 ──')
    print(f'{"Tag":<28}  {"BLEU-4":>7}  {"time(s)":>8}  {"#params":>10}')
    print('─' * 60)
    for r in results_sorted:
        print(f'  {r["tag"]:<26}  {r["test_bleu4"]:>7.4f}  '
              f'{r["train_time"]:>8.0f}  {r["n_params"]:>10,}')


if __name__ == '__main__':
    main()
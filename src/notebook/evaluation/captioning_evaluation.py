import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import tensorflow as tf
from tensorflow import keras
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

from algorithm.core.diffable    import Diffable
from algorithm.core.parameter   import Parameter, Input
from algorithm.function.relu    import ReLU
from algorithm.function.softmax import Softmax
from algorithm.neural.embedding import Embedding
from algorithm.neural.rnn       import SimpleRNN
from algorithm.neural.lstm      import LSTM
from algorithm.neural.linear    import Linear
from algorithm.neural.concat    import Concatenate as ConcatNode

PROC_DIR   = REPO_ROOT / 'data_processed'
MODELS_DIR = REPO_ROOT / 'models_captioning'

def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def _relu(x):
    return np.maximum(0.0, x)

class ScratchDecoder:

    def __init__(self, cell_type: str, weights: dict, vocab: dict,
                 embed_dim: int, units_per_layer: list):
        self.cell_type = cell_type          # 'rnn' or 'lstm'
        self.w         = weights
        self.vocab     = vocab
        self.id2word   = {v: k for k, v in vocab.items()}
        self.embed_dim = embed_dim
        self.n_layers  = len(units_per_layer)
        self.units     = units_per_layer    

    def _rnn_step(self, x, h, layer_idx):
        W_x = self.w[f'rnn_{layer_idx}_Wx']
        W_h = self.w[f'rnn_{layer_idx}_Wh']
        b   = self.w[f'rnn_{layer_idx}_b']
        return np.tanh(x @ W_x + h @ W_h + b)

    def _lstm_step(self, x, h, c, layer_idx):
        W = self.w[f'lstm_{layer_idx}_W']
        U = self.w[f'lstm_{layer_idx}_U']
        b = self.w[f'lstm_{layer_idx}_b']
        if b.ndim == 2:
            b = b.sum(axis=0)
        u = U.shape[0]
        z = x @ W + h @ U + b
        i_t = _sigmoid(z[:u]);       f_t = _sigmoid(z[u:2*u])
        g_t = np.tanh(z[2*u:3*u]);   o_t = _sigmoid(z[3*u:])
        c   = f_t * c + i_t * g_t
        h   = o_t * np.tanh(c)
        return h, c

    def generate(self, img_feat: np.ndarray, max_len: int = 35) -> list:
        """Generate a caption for a single image feature vector.

        Args:
            img_feat: (feature_dim,) — CNN feature for one image.
            max_len:  maximum number of tokens to generate.

        Returns:
            List of word strings (without <start>/<end>).
        """
        img_proj = _relu(img_feat @ self.w['img_proj_W'] + self.w['img_proj_b'])

        hs = [np.zeros(u) for u in self.units]
        cs = [np.zeros(u) for u in self.units]

        x_t    = img_proj
        tokens = []
        end_id = self.vocab['<end>']
        pad_id = self.vocab['<pad>']
        E      = self.w['embedding']

        for _ in range(max_len):
            for li in range(self.n_layers):
                if self.cell_type == 'rnn':
                    hs[li] = self._rnn_step(x_t, hs[li], li)
                    x_t = hs[li]
                else:
                    hs[li], cs[li] = self._lstm_step(x_t, hs[li], cs[li], li)
                    x_t = hs[li]

            logits = x_t @ self.w['output_W'] + self.w['output_b']
            probs  = _softmax(logits)
            tok    = int(np.argmax(probs))

            if tok == end_id or tok == pad_id:
                break
            tokens.append(tok)
            x_t = E[tok]

        return [self.id2word.get(t, '<unk>') for t in tokens]

def extract_weights(keras_model: keras.Model, cell_type: str, n_layers: int) -> dict:
    weights = {}
    for layer in keras_model.layers:
        name = layer.name
        lw   = layer.get_weights()
        if not lw:
            continue
        if name == 'img_proj':
            weights['img_proj_W'], weights['img_proj_b'] = lw
        elif name == 'embedding':
            weights['embedding'] = lw[0]
        elif name == 'output':
            weights['output_W'], weights['output_b'] = lw
        elif cell_type == 'lstm' and name.startswith('lstm_'):
            idx = int(name.split('_')[1])
            weights[f'lstm_{idx}_W'], weights[f'lstm_{idx}_U'], weights[f'lstm_{idx}_b'] = lw
        elif cell_type == 'rnn' and name.startswith('rnn_'):
            idx = int(name.split('_')[1])
            weights[f'rnn_{idx}_Wx'], weights[f'rnn_{idx}_Wh'], weights[f'rnn_{idx}_b'] = lw
    return weights

def build_diffable_decoder(weights: dict, cell_type: str, n_layers: int,
                           units_list: list, max_len: int, vocab_size: int):
    """Build a Diffable graph for one forward pass (teacher forcing, batch mode).

    Returns (img_inp, cap_inp, output_node).
    """
    img_inp = Input()
    cap_inp = Input()

    W_proj  = Parameter(weights['img_proj_W'])
    b_proj  = Parameter(weights['img_proj_b'])
    img_p   = ReLU(Linear(img_inp, W_proj, b_proj))
    img_seq = _ExpandDim(img_p)

    E_mat  = Parameter(weights['embedding'])
    cap_e  = Embedding(cap_inp, E_mat)

    seq = ConcatNode(img_seq, cap_e, axis=1)
    x = seq
    for li in range(n_layers):
        if cell_type == 'lstm':
            W = Parameter(weights[f'lstm_{li}_W'])
            U = Parameter(weights[f'lstm_{li}_U'])
            b = Parameter(weights[f'lstm_{li}_b'])
            x = LSTM(x, W, U, b, return_sequences=True)
        else:
            Wx = Parameter(weights[f'rnn_{li}_Wx'])
            Wh = Parameter(weights[f'rnn_{li}_Wh'])
            b  = Parameter(weights[f'rnn_{li}_b'])
            x  = SimpleRNN(x, Wx, Wh, b, return_sequences=True)

    x_sliced = _SliceSeq(x, max_len)

    W_out = Parameter(weights['output_W'])
    b_out = Parameter(weights['output_b'])
    out   = _SeqLinear(x_sliced, W_out, b_out)

    return img_inp, cap_inp, out

class _ExpandDim(Diffable):
    """Unsqueeze axis=1: (N, D) → (N, 1, D)."""
    def __init__(self, x: Diffable):
        super().__init__(x)

    def _calculate_value(self, s: dict) -> np.ndarray:
        return list(s.values())[0][:, np.newaxis, :]

    def _calculate_gradient(self, s: dict, v: np.ndarray) -> dict:
        return {list(s.keys())[0]: v[:, 0, :]}


class _SliceSeq(Diffable):
    """Slice first `length` timesteps: (N, T+1, U) → (N, T, U)."""
    _warn_on_unmanaged_state = False

    def __init__(self, x: Diffable, length: int):
        self._len = length
        super().__init__(x)

    def _calculate_value(self, s: dict) -> np.ndarray:
        return list(s.values())[0][:, :self._len, :]

    def _calculate_gradient(self, s: dict, v: np.ndarray) -> dict:
        k, x = next(iter(s.items()))
        g = np.zeros_like(x)
        g[:, :self._len, :] = v
        return {k: g}


class _SeqLinear(Diffable):
    """Dense applied to every timestep: (N, T, in) @ (in, out) + (out,) → (N, T, out)."""
    def __init__(self, x: Diffable, W: Diffable, b: Diffable):
        super().__init__(x, W, b)

    def _calculate_value(self, s: dict) -> np.ndarray:
        x, W, b = list(s.values())
        return x @ W + b

    def _calculate_gradient(self, s: dict, v: np.ndarray) -> dict:
        x_n, W_n, b_n = list(s.keys())
        x, W, _ = list(s.values())
        return {x_n: v @ W.T,
                W_n: x.reshape(-1, x.shape[-1]).T @ v.reshape(-1, v.shape[-1]),
                b_n: v.reshape(-1, v.shape[-1]).sum(axis=0)}

def evaluate_corpus(hypotheses: list, ref_map: dict, img_order: list) -> dict:
    """Compute BLEU-4 and METEOR for a list of hypotheses."""
    smooth = SmoothingFunction().method1

    refs_for_bleu = [[r.split() for r in ref_map.get(img, ['a'])]
                     for img in img_order]
    bleu4 = corpus_bleu(refs_for_bleu, hypotheses,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smooth)

    meteor_scores = []
    for hyp, img in zip(hypotheses, img_order):
        refs = [r.split() for r in ref_map.get(img, ['a'])]
        meteor_scores.append(meteor_score(refs, hyp))
    meteor = float(np.mean(meteor_scores))

    return {'bleu4': float(bleu4), 'meteor': meteor}



def main():
    test_img = np.load(PROC_DIR / 'test_img_feats.npy')

    with open(PROC_DIR / 'vocab.json') as f:
        vocab = json.load(f)
    with open(PROC_DIR / 'test_references.json') as f:
        ref_map = json.load(f)

    test_imgs_order = sorted(ref_map.keys())
    id2word   = {v: k for k, v in vocab.items()}
    embed_dim = 256  # must match training
    max_len   = 35

    with open(MODELS_DIR / 'captioning_results.json') as f:
        train_results = json.load(f)

    print('\n══ A. All 12 variants (from-scratch greedy decode) ══')
    print(f'{"Tag":<28}  {"BLEU-4":>7}  {"METEOR":>7}  {"time(s)":>8}')
    print('─' * 60)

    scratch_results = []
    for r in sorted(train_results, key=lambda x: (x['cell_type'], x['n_layers'], x['units'])):
        tag        = r['tag']
        cell_type  = r['cell_type']
        n_layers   = r['n_layers']
        units      = r['units']

        keras_model = keras.models.load_model(r['saved_to'])
        weights     = extract_weights(keras_model, cell_type, n_layers)
        units_list  = [units] * n_layers
        decoder     = ScratchDecoder(cell_type, weights, vocab, embed_dim, units_list)

        t0   = time.time()
        hyps = [decoder.generate(test_img[i], max_len) for i in range(len(test_img))]
        dt   = time.time() - t0

        metrics = evaluate_corpus(hyps, ref_map, test_imgs_order)
        print(f'  {tag:<26}  {metrics["bleu4"]:>7.4f}  '
              f'{metrics["meteor"]:>7.4f}  {dt:>8.1f}')
        scratch_results.append({'tag': tag, 'cell_type': cell_type,
                                'n_layers': n_layers, 'units': units,
                                **metrics, 'scratch_time': dt,
                                'keras_bleu4': r['test_bleu4']})

    print('\n══ B. Keras vs From-Scratch (best per cell type) ══')
    for cell in ['rnn', 'lstm']:
        cell_res = [r for r in scratch_results if r['cell_type'] == cell]
        best_scratch = max(cell_res, key=lambda x: x['bleu4'])
        best_keras   = max(cell_res, key=lambda x: x['keras_bleu4'])
        print(f'\n  {cell.upper()} — best from-scratch: {best_scratch["tag"]}')
        print(f'    Scratch BLEU-4 = {best_scratch["bleu4"]:.4f}  '
              f'METEOR = {best_scratch["meteor"]:.4f}  '
              f'time = {best_scratch["scratch_time"]:.1f}s')
        print(f'    Keras   BLEU-4 = {best_keras["keras_bleu4"]:.4f}')
        diff = abs(best_scratch['bleu4'] - best_keras['keras_bleu4'])
        print(f'    |diff| BLEU-4  = {diff:.4f}')

    print('\n══ C. RNN vs LSTM: qualitative analysis ══')
    best_rnn  = max([r for r in scratch_results if r['cell_type'] == 'rnn'],
                    key=lambda x: x['bleu4'])
    best_lstm = max([r for r in scratch_results if r['cell_type'] == 'lstm'],
                    key=lambda x: x['bleu4'])

    rnn_model  = keras.models.load_model(next(r['saved_to'] for r in train_results
                                              if r['tag'] == best_rnn['tag']))
    lstm_model = keras.models.load_model(next(r['saved_to'] for r in train_results
                                              if r['tag'] == best_lstm['tag']))

    rnn_w  = extract_weights(rnn_model,  'rnn',  best_rnn['n_layers'])
    lstm_w = extract_weights(lstm_model, 'lstm', best_lstm['n_layers'])

    rnn_dec  = ScratchDecoder('rnn',  rnn_w,  vocab, embed_dim,
                              [best_rnn['units']]  * best_rnn['n_layers'])
    lstm_dec = ScratchDecoder('lstm', lstm_w, vocab, embed_dim,
                              [best_lstm['units']] * best_lstm['n_layers'])

    print(f'\n  {"#":<4}  {"RNN caption":<45}  {"LSTM caption"}')
    print('  ' + '─' * 100)
    sample_idxs = list(range(0, min(10, len(test_img))))
    for i, idx in enumerate(sample_idxs):
        rnn_cap  = ' '.join(rnn_dec.generate( test_img[idx], max_len))
        lstm_cap = ' '.join(lstm_dec.generate(test_img[idx], max_len))
        img_name = test_imgs_order[idx] if idx < len(test_imgs_order) else f'img_{idx}'
        print(f'  {i+1:<4}  {rnn_cap[:44]:<45}  {lstm_cap[:44]}')
        refs = ref_map.get(img_name, ['(no reference)'])
        print(f'        REF: {refs[0][:90]}')

    print('\n══ D. Max caption length sweep ══')
    best_overall = max(scratch_results, key=lambda x: x['bleu4'])
    tag_best     = best_overall['tag']
    cell_best    = best_overall['cell_type']
    n_layers_best= best_overall['n_layers']
    units_best   = best_overall['units']

    model_best = keras.models.load_model(next(r['saved_to'] for r in train_results
                                              if r['tag'] == tag_best))
    weights_best = extract_weights(model_best, cell_best, n_layers_best)
    dec_best = ScratchDecoder(cell_best, weights_best, vocab, embed_dim,
                              [units_best] * n_layers_best)

    print(f'  Best model: {tag_best}')
    print(f'  {"max_len":<10}  {"BLEU-4":>7}  {"METEOR":>7}')
    print('  ' + '─' * 30)

    for ml in [15, 25, 35, 50]:
        hyps_ml = [dec_best.generate(test_img[i], ml) for i in range(len(test_img))]
        m = evaluate_corpus(hyps_ml, ref_map, test_imgs_order)
        print(f'  {ml:<10}  {m["bleu4"]:>7.4f}  {m["meteor"]:>7.4f}')

    print('\n══ Final Summary ══')
    print(f'{"Tag":<28}  {"Scratch BLEU4":>13}  {"Keras BLEU4":>11}  {"METEOR":>7}  {"time(s)":>8}')
    print('─' * 80)
    for r in sorted(scratch_results, key=lambda x: x['bleu4'], reverse=True):
        print(f'  {r["tag"]:<26}  {r["bleu4"]:>13.4f}  '
              f'{r["keras_bleu4"]:>11.4f}  {r["meteor"]:>7.4f}  '
              f'{r["scratch_time"]:>8.1f}')

    with open(MODELS_DIR / 'evaluation_results.json', 'w') as f:
        json.dump(scratch_results, f, indent=2)
    print('\nResults saved → evaluation_results.json')


if __name__ == '__main__':
    main()

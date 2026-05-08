"""
CNN Evaluation — From-Scratch Forward Propagation (IF3270 Tubes 2)
===================================================================
Loads the best Keras model, rebuilds it from scratch using our NumPy
Diffable nodes (shared Conv2D and non-shared LocallyConnected2D), then
compares macro F1-scores on the test set.

Run from the repo root:
    python src/notebook/cnn_evaluation.py

Prerequisites:
    - cnn_training.py has been run → models/ directory with .keras files
    - dataset/seg_test/ exists
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, classification_report

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import tensorflow as tf
from tensorflow import keras

from algorithm.core.parameter import Parameter, Input
from algorithm.function.relu    import ReLU
from algorithm.function.softmax import Softmax
from algorithm.neural.conv      import Conv2D
from algorithm.neural.local     import LocallyConnected2D
from algorithm.neural.maxpool   import MaxPool2D
from algorithm.neural.avgpool   import AvgPool2D
from algorithm.neural.reshape   import Flatten
from algorithm.neural.linear    import Linear
from algorithm.utility.image_utils import batch_loader

# ─── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = REPO_ROOT / 'models'
DATASET_DIR = REPO_ROOT / 'dataset'
TEST_DIR    = DATASET_DIR / 'seg_test' / 'seg_test'

IMG_SIZE    = (150, 150)
BATCH_SIZE  = 16          # smaller for from-scratch (NumPy is slower)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_test_data(test_dir: Path, img_size: tuple = IMG_SIZE):
    """Return (images_array, labels_array) for the test set."""
    from PIL import Image
    images, labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob('*.jpg')):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_size[1], img_size[0]))
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(label_idx)
    return np.stack(images), np.array(labels, dtype=np.int32)


# ─── From-scratch model builder ───────────────────────────────────────────────

def build_scratch_model(keras_model: keras.Model,
                        use_locally_connected: bool = False):
    """Mirror a Keras CNN model using Diffable nodes.

    Returns:
        (input_node, output_node): the endpoints of the Diffable graph.
    """
    input_node = Input()
    current = input_node

    # For LocallyConnected2D we need output shapes to tile shared weights.
    # Obtain them via a dummy forward pass in Keras.
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    layer_outputs = {}
    if use_locally_connected:
        for layer in keras_model.layers:
            if layer.name != keras_model.layers[0].name:  # skip InputLayer
                sub = keras.Model(inputs=keras_model.input, outputs=layer.output)
                layer_outputs[layer.name] = sub.predict(dummy, verbose=0).shape

    for layer in keras_model.layers:
        ltype = type(layer).__name__
        weights = layer.get_weights()
        cfg = layer.get_config()

        if ltype in ('InputLayer', 'Dropout'):
            continue

        elif ltype == 'Conv2D':
            kernel_np, bias_np = weights
            strides = tuple(cfg['strides'])
            padding = cfg['padding']

            if use_locally_connected:
                out_shape = layer_outputs[layer.name]   # (1, oH, oW, Cout)
                oH, oW = out_shape[1], out_shape[2]
                n_pos = oH * oW
                kH, kW, C_in, C_out = kernel_np.shape
                # Tile shared kernel → per-position kernel
                lc_kernel = np.tile(
                    kernel_np.reshape(1, kH * kW * C_in, C_out),
                    (n_pos, 1, 1),
                )
                lc_bias = np.tile(bias_np.reshape(1, C_out), (n_pos, 1))
                k_node = Parameter(lc_kernel)
                b_node = Parameter(lc_bias)
                current = LocallyConnected2D(
                    current, k_node, b_node,
                    kernel_size=(kH, kW),
                    strides=strides,
                    padding=padding,
                )
            else:
                k_node = Parameter(kernel_np)
                b_node = Parameter(bias_np)
                current = Conv2D(current, k_node, b_node,
                                 strides=strides, padding=padding)

            activation = cfg.get('activation', 'linear')
            if activation == 'relu':
                current = ReLU(current)
            elif activation == 'softmax':
                current = Softmax(current)

        elif ltype in ('MaxPooling2D',):
            current = MaxPool2D(current,
                                pool_size=tuple(cfg['pool_size']),
                                strides=tuple(cfg['strides']),
                                padding=cfg['padding'])

        elif ltype in ('AveragePooling2D',):
            current = AvgPool2D(current,
                                pool_size=tuple(cfg['pool_size']),
                                strides=tuple(cfg['strides']),
                                padding=cfg['padding'])

        elif ltype == 'Flatten':
            current = Flatten(current)

        elif ltype == 'Dense':
            kernel_np, bias_np = weights
            k_node = Parameter(kernel_np)
            b_node = Parameter(bias_np)
            current = Linear(current, k_node, b_node)
            activation = cfg.get('activation', 'linear')
            if activation == 'relu':
                current = ReLU(current)
            elif activation == 'softmax':
                current = Softmax(current)

        elif ltype == 'BatchNormalization':
            # BatchNorm in inference mode: (x - mean) / sqrt(var + eps) * gamma + beta
            # Applied as a fixed affine transform using trained running statistics.
            gamma, beta, mean, var = weights
            epsilon = cfg.get('epsilon', 1e-3)
            scale = gamma / np.sqrt(var + epsilon)
            # Wrap as a Lambda-style node using Linear with diagonal weight
            # For a (N, ..., C) input this is equivalent to channel-wise scaling + bias
            # We handle it by applying the transform to the Parameter directly.
            # Simple approach: absorb BN into a single affine Parameter node.
            # (full BN node not implemented; skip if not present in training arch)
            print(f"  [WARN] BatchNormalization layer '{layer.name}' not fully "
                  f"supported in from-scratch mode — skipping.")
            continue

    return input_node, current


# ─── Inference runner ─────────────────────────────────────────────────────────

def predict_scratch(input_node: Input, output_node,
                    x_test: np.ndarray) -> np.ndarray:
    """Run batch-wise forward pass through the Diffable graph."""
    preds = []
    for start in range(0, len(x_test), BATCH_SIZE):
        batch = x_test[start:start + BATCH_SIZE]
        # Clear cached values (Parameter nodes keep their weights)
        input_node.clear_values()
        input_node._value = batch
        out = output_node.get_value()       # (batch, C)
        preds.append(np.argmax(out, axis=1))
    return np.concatenate(preds)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Find best model from training results
    results_path = MODELS_DIR / 'training_results.json'
    if not results_path.exists():
        print(f"[ERROR] {results_path} not found. Run cnn_training.py first.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    results_sorted = sorted(results, key=lambda r: r['test_macro_f1'], reverse=True)
    best = results_sorted[0]
    print(f"Best model (test macro-F1 = {best['test_macro_f1']:.4f}): {best['tag']}")

    keras_model = keras.models.load_model(best['saved_to'])
    keras_model.summary()

    # 2. Load test data
    print("\nLoading test data …")
    x_test, y_test = load_test_data(TEST_DIR)
    print(f"  {len(x_test)} test images loaded.")

    # 3. Keras baseline predictions
    print("\n[Keras] Running inference …")
    keras_probs = keras_model.predict(x_test, batch_size=32, verbose=1)
    keras_preds = np.argmax(keras_probs, axis=1)
    keras_f1    = f1_score(y_test, keras_preds, average='macro')
    print(f"  Keras macro F1: {keras_f1:.4f}")
    print(classification_report(y_test, keras_preds, target_names=CLASS_NAMES))

    # 4. From-scratch — shared (Conv2D)
    print("\n[From-scratch / Conv2D] Building graph …")
    inp_shared, out_shared = build_scratch_model(keras_model, use_locally_connected=False)
    print("[From-scratch / Conv2D] Running inference …")
    scratch_shared_preds = predict_scratch(inp_shared, out_shared, x_test)
    scratch_shared_f1    = f1_score(y_test, scratch_shared_preds, average='macro')
    print(f"  From-scratch (Conv2D) macro F1: {scratch_shared_f1:.4f}")
    print(classification_report(y_test, scratch_shared_preds, target_names=CLASS_NAMES))

    # 5. From-scratch — non-shared (LocallyConnected2D)
    print("\n[From-scratch / LocallyConnected2D] Building graph …")
    inp_lc, out_lc = build_scratch_model(keras_model, use_locally_connected=True)
    print("[From-scratch / LocallyConnected2D] Running inference …")
    scratch_lc_preds = predict_scratch(inp_lc, out_lc, x_test)
    scratch_lc_f1    = f1_score(y_test, scratch_lc_preds, average='macro')
    print(f"  From-scratch (LocallyConnected2D) macro F1: {scratch_lc_f1:.4f}")
    print(classification_report(y_test, scratch_lc_preds, target_names=CLASS_NAMES))

    # 6. Summary
    print("\n══ Summary ══")
    print(f"  Keras                 macro F1 = {keras_f1:.4f}")
    print(f"  From-scratch Conv2D   macro F1 = {scratch_shared_f1:.4f}  "
          f"(diff = {abs(keras_f1 - scratch_shared_f1):.6f})")
    print(f"  From-scratch LC2D     macro F1 = {scratch_lc_f1:.4f}  "
          f"(diff = {abs(keras_f1 - scratch_lc_f1):.6f})")

    # 7. All 16 variants summary (from training results)
    print("\n══ All 16 variants ══")
    print(f"{'Tag':<42}  {'val F1':>8}  {'test F1':>8}  {'params':>10}")
    print('-' * 75)
    for r in results_sorted:
        print(f"  {r['tag']:<40}  {r['val_macro_f1']:>8.4f}  "
              f"{r['test_macro_f1']:>8.4f}  {r['total_params']:>10,}")


if __name__ == '__main__':
    main()

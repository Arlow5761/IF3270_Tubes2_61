"""
CNN Evaluation — From-Scratch Forward Propagation (IF3270 Tubes 2)
===================================================================
Loads the best Keras model, rebuilds it from scratch using our NumPy
Diffable nodes, then compares macro F1-scores on the test set.

Run from the repo root:
    python src/notebook/evaluation/cnn_evaluation.py

Prerequisites:
    - cnn_training.py has been run → models/ directory with .keras files
    - dataset/seg_test/ exists
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, classification_report

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import tensorflow as tf
from tensorflow import keras

from algorithm.core.parameter import Parameter, Input
from algorithm.function.relu    import ReLU
from algorithm.function.softmax import Softmax
from algorithm.neural.conv      import Conv2D
# Renamed to avoid collision with Keras Layer
from algorithm.neural.local     import LocallyConnected2D as DiffableLocallyConnected2D
from algorithm.neural.maxpool   import MaxPool2D
from algorithm.neural.avgpool   import AvgPool2D
from algorithm.neural.reshape   import Flatten
from algorithm.neural.linear    import Linear
from algorithm.utility.image_utils import batch_loader

# Import to ensure keras.models.load_model successfully resolves the custom layer
from compat.LocallyConnected2D import LocallyConnected2D as KerasLocallyConnected2D

MODELS_DIR = REPO_ROOT / 'models'
DATASET_DIR = REPO_ROOT / 'dataset'
TEST_DIR    = DATASET_DIR / 'seg_test' / 'seg_test'

IMG_SIZE    = (150, 150)
BATCH_SIZE  = 16          # smaller for from-scratch (NumPy is slower)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


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


def build_scratch_model(keras_model: keras.Model):
    """Mirror a Keras CNN model using Diffable nodes natively."""
    input_node = Input()
    current = input_node

    for layer in keras_model.layers:
        ltype = type(layer).__name__
        weights = layer.get_weights()
        cfg = layer.get_config()

        if ltype in ('InputLayer', 'Dropout'):
            continue

        elif ltype == 'Conv2D':
            kernel_np, bias_np = weights
            k_node = Parameter(kernel_np)
            b_node = Parameter(bias_np)
            
            current = Conv2D(current, k_node, b_node,
                             strides=tuple(cfg['strides']), 
                             padding=cfg['padding'])

            activation = cfg.get('activation', 'linear')
            if isinstance(activation, dict):
                activation = activation.get('config', activation.get('name', 'linear'))
            if isinstance(activation, str):
                if activation.lower() == 'relu':
                    current = ReLU(current)
                elif activation.lower() == 'softmax':
                    current = Softmax(current)

        elif ltype == 'LocallyConnected2D':
            kernel_np, bias_np = weights
            
            # Keras custom weights: (out_H, out_W, kH*kW*C_in, C_out)
            # Diffable node expected: (out_H*out_W, kH*kW*C_in, C_out)
            out_H, out_W, klen, C_out = kernel_np.shape
            lc_kernel = kernel_np.reshape(out_H * out_W, klen, C_out)
            lc_bias = bias_np.reshape(out_H * out_W, C_out)

            k_node = Parameter(lc_kernel)
            b_node = Parameter(lc_bias)

            current = DiffableLocallyConnected2D(
                current, k_node, b_node,
                kernel_size=tuple(cfg['kernel_size']),
                strides=(1, 1),    # custom keras implementation uses hardcoded strides=(1,1)
                padding='valid',   # custom keras implementation uses hardcoded padding='valid'
            )

            activation = cfg.get('activation', 'linear')
            if isinstance(activation, dict):
                activation = activation.get('config', activation.get('name', 'linear'))
            if isinstance(activation, str):
                if activation.lower() == 'relu':
                    current = ReLU(current)
                elif activation.lower() == 'softmax':
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
            if isinstance(activation, dict):
                activation = activation.get('config', activation.get('name', 'linear'))
            if isinstance(activation, str):
                if activation.lower() == 'relu':
                    current = ReLU(current)
                elif activation.lower() == 'softmax':
                    current = Softmax(current)

        elif ltype == 'BatchNormalization':
            print(f"  [WARN] BatchNormalization layer '{layer.name}' not supported — skipping.")
            continue

    return input_node, current


def predict_scratch(input_node: Input, output_node,
                    x_test: np.ndarray) -> np.ndarray:
    """Run batch-wise forward pass through the Diffable graph."""
    preds = []
    for start in range(0, len(x_test), BATCH_SIZE):
        batch = x_test[start:start + BATCH_SIZE]
        input_node.clear_values()
        input_node._value = batch
        out = output_node.get_value()
        preds.append(np.argmax(out, axis=1))
    return np.concatenate(preds)


def main():
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

    print("\nLoading test data …")
    x_test, y_test = load_test_data(TEST_DIR)
    print(f"  {len(x_test)} test images loaded.")

    print("\n[Keras] Running inference …")
    keras_probs = keras_model.predict(x_test, batch_size=32, verbose=1)
    keras_preds = np.argmax(keras_probs, axis=1)
    keras_f1    = f1_score(y_test, keras_preds, average='macro')
    print(f"  Keras macro F1: {keras_f1:.4f}")
    print(classification_report(y_test, keras_preds, target_names=CLASS_NAMES))

    print("\n[From-scratch] Building graph …")
    inp_scratch, out_scratch = build_scratch_model(keras_model)
    
    print("[From-scratch] Running inference …")
    scratch_preds = predict_scratch(inp_scratch, out_scratch, x_test)
    scratch_f1    = f1_score(y_test, scratch_preds, average='macro')
    print(f"  From-scratch macro F1: {scratch_f1:.4f}")
    print(classification_report(y_test, scratch_preds, target_names=CLASS_NAMES))

    print("\n══ Summary ══")
    print(f"  Keras           macro F1 = {keras_f1:.4f}")
    print(f"  From-scratch    macro F1 = {scratch_f1:.4f}  "
          f"(diff = {abs(keras_f1 - scratch_f1):.6f})")

    print("\n══ All 16 variants ══")
    print(f"{'Tag':<42}  {'val F1':>8}  {'test F1':>8}  {'params':>10}")
    print('-' * 75)
    for r in results_sorted:
        print(f"  {r['tag']:<40}  {r['val_macro_f1']:>8.4f}  "
              f"{r['test_macro_f1']:>8.4f}  {r['total_params']:>10,}")


if __name__ == '__main__':
    main()
"""
CNN Training — Intel Image Classification (IF3270 Tubes 2)
==========================================================
Trains 16 CNN architecture variants and saves each model.
Run from the repo root:
    python src/notebook/cnn_training.py

Dataset expected at:  dataset/intel-image-classification
  dataset/train/   → sub-folders per class
  dataset/val/     → sub-folders per class
  dataset/test/    → sub-folders per class (or a flat test set)

Classes: buildings, forest, glacier, mountain, sea, street
"""

import os
import json
import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score

REPO_ROOT   = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / 'dataset' / 'intel-image-classification'
MODELS_DIR  = REPO_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_DIR = DATASET_DIR / 'seg_train' / 'seg_train'
VAL_DIR   = DATASET_DIR / 'seg_test'  / 'seg_test'   # used as validation
TEST_DIR  = DATASET_DIR / 'seg_test'  / 'seg_test'

IMG_SIZE   = (150, 150)
N_CLASSES  = 6
BATCH_SIZE = 32
EPOCHS     = 20
SEED       = 42

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


def make_dataset(directory: Path, shuffle: bool = False) -> tf.data.Dataset:
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(directory),
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
    normalization = layers.Rescaling(1.0 / 255)
    return ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
             .prefetch(tf.data.AUTOTUNE)



def build_model(n_blocks: int, filters: list, kernel_size: tuple,
                pool_type: str) -> keras.Model:
    """Build a CNN with `n_blocks` conv-pool blocks.

    Args:
        n_blocks:    Number of Conv+Pool blocks (2 or 3).
        filters:     List of filter counts, one per block.
        kernel_size: (height, width) of conv kernels.
        pool_type:   'max' or 'avg'.
    """
    assert len(filters) == n_blocks, "filters list length must equal n_blocks"
    pool_layer = layers.MaxPooling2D if pool_type == 'max' else layers.AveragePooling2D

    model = keras.Sequential(name=f"cnn_b{n_blocks}_k{kernel_size[0]}_p{pool_type}")
    model.add(layers.Input(shape=(*IMG_SIZE, 3)))

    for i in range(n_blocks):
        model.add(layers.Conv2D(filters[i], kernel_size, padding='same', activation='relu'))
        model.add(pool_layer(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(N_CLASSES, activation='softmax'))
    return model



# 2 blocks × 2 filter sizes × 2 kernel sizes × 2 pool types = 16 variants
EXPERIMENTS = []
for n_blocks, filters_key, kernel_size, pool_type in itertools.product(
    [2, 3],
    ['small', 'large'],
    [(3, 3), (5, 5)],
    ['max', 'avg'],
):
    filters_map = {
        (2, 'small'): [32, 64],
        (2, 'large'): [64, 128],
        (3, 'small'): [32, 64, 128],
        (3, 'large'): [64, 128, 256],
    }
    filters = filters_map[(n_blocks, filters_key)]
    EXPERIMENTS.append({
        'n_blocks':    n_blocks,
        'filters':     filters,
        'filters_key': filters_key,
        'kernel_size': kernel_size,
        'pool_type':   pool_type,
    })

assert len(EXPERIMENTS) == 16, f"Expected 16 experiments, got {len(EXPERIMENTS)}"



def evaluate_model(model: keras.Model, dataset: tf.data.Dataset) -> dict:
    y_true, y_pred = [], []
    for xb, yb in dataset:
        probs = model.predict(xb, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(yb.numpy().tolist())
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return {'macro_f1': float(macro_f1), 'n_samples': len(y_true)}



def main():
    print(f"TensorFlow {tf.__version__}  |  GPU: {tf.config.list_physical_devices('GPU')}")

    train_ds = make_dataset(TRAIN_DIR, shuffle=True)
    val_ds   = make_dataset(VAL_DIR,   shuffle=False)
    test_ds  = make_dataset(TEST_DIR,  shuffle=False)

    results = []

    for idx, cfg in enumerate(EXPERIMENTS, 1):
        tag = (f"b{cfg['n_blocks']}_f{cfg['filters_key']}"
               f"_k{cfg['kernel_size'][0]}_p{cfg['pool_type']}")
        print(f"\n[{idx:02d}/16]  {tag}  filters={cfg['filters']}")

        model = build_model(
            n_blocks    = cfg['n_blocks'],
            filters     = cfg['filters'],
            kernel_size = cfg['kernel_size'],
            pool_type   = cfg['pool_type'],
        )
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=1e-3),
            loss      = keras.losses.SparseCategoricalCrossentropy(),
            metrics   = ['accuracy'],
        )
        model.summary(print_fn=lambda s: None)  # suppress verbose output

        history = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs          = EPOCHS,
            callbacks       = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            ],
            verbose=1,
        )

        val_metrics  = evaluate_model(model, val_ds)
        test_metrics = evaluate_model(model, test_ds)

        save_path = MODELS_DIR / f'cnn_{tag}.keras'
        model.save(str(save_path))

        result = {
            'tag':            tag,
            'config':         {k: (list(v) if isinstance(v, (list, tuple)) else v)
                               for k, v in cfg.items()},
            'train_loss':     history.history['loss'],
            'val_loss':       history.history['val_loss'],
            'train_acc':      history.history['accuracy'],
            'val_acc':        history.history['val_accuracy'],
            'val_macro_f1':   val_metrics['macro_f1'],
            'test_macro_f1':  test_metrics['macro_f1'],
            'saved_to':       str(save_path),
            'total_params':   model.count_params(),
        }
        results.append(result)
        print(f"  val  macro-F1 = {val_metrics['macro_f1']:.4f}")
        print(f"  test macro-F1 = {test_metrics['macro_f1']:.4f}")

    summary_path = MODELS_DIR / 'training_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {summary_path}")

    results_sorted = sorted(results, key=lambda r: r['test_macro_f1'], reverse=True)
    print("\n── Ranking by test macro F1 ──")
    for rank, r in enumerate(results_sorted, 1):
        print(f"  #{rank:02d}  {r['tag']:<40}  {r['test_macro_f1']:.4f}")

    best = results_sorted[0]
    print(f"\nBest model: {best['tag']}  (test macro-F1 = {best['test_macro_f1']:.4f})")
    print(f"Saved at:   {best['saved_to']}")


if __name__ == '__main__':
    main()

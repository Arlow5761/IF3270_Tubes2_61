from __future__ import annotations
from pathlib import Path
from typing import List, Union
import numpy as np


def image_loader(path: Union[str, Path], target_size: tuple = (150, 150)) -> np.ndarray:
    """Load a single image and return as float32 numpy array (H, W, C) in [0, 1].

    Args:
        path: Path to the image file.
        target_size: (height, width) to resize the image to.

    Returns:
        Array of shape (H, W, 3), dtype float32, values in [0, 1].
    """
    from PIL import Image
    img = Image.open(path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)  # PIL uses (W, H)
    return np.array(img, dtype=np.float32) / 255.0


def batch_loader(path_list: List[Union[str, Path]],
                 target_size: tuple = (150, 150)) -> np.ndarray:
    """Load a list of images into a single batch array.

    Args:
        path_list: List of image file paths.
        target_size: (height, width) for each image.

    Returns:
        Array of shape (N, H, W, 3), dtype float32, values in [0, 1].
    """
    images = [image_loader(p, target_size) for p in path_list]
    return np.stack(images, axis=0)


def feature_extractor(path_list: List[Union[str, Path]],
                      keras_encoder,
                      target_size: tuple = (150, 150),
                      output_path: Union[str, Path] = 'features.npy',
                      batch_size: int = 32) -> np.ndarray:
    """Extract features from images using a Keras encoder and save to .npy.

    Args:
        path_list:      List of image file paths.
        keras_encoder:  A Keras model (encoder portion) whose predict() is called.
        target_size:    (height, width) used when loading each image.
        output_path:    Destination .npy file for the extracted features.
        batch_size:     Number of images processed per Keras predict() call.

    Returns:
        features: Array of shape (N, *feature_shape).
    """
    all_features = []
    for start in range(0, len(path_list), batch_size):
        batch_paths = path_list[start:start + batch_size]
        batch = batch_loader(batch_paths, target_size)
        feats = keras_encoder.predict(batch, verbose=0)
        all_features.append(feats)
    features = np.concatenate(all_features, axis=0)
    np.save(output_path, features)
    return features

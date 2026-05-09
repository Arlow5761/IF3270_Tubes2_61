"""
Image Captioning — Preprocessing: Feature Extraction + Vocabulary (IF3270 Tubes 2)
====================================================================================
Run once to prepare everything needed before training.

Expected Flickr8k dataset structure (Kaggle format):
  dataset/flickr8k/Images/           ← all 8092 JPEG images
  dataset/flickr8k/captions.txt      ← "image_name,caption" CSV (one header line)

OR the older Flickr8k text format:
  dataset/flickr8k/Flickr8k.token.txt  ← "img.jpg#i\\tcaption text"
  dataset/flickr8k/Flickr_8k.trainImages.txt
  dataset/flickr8k/Flickr_8k.devImages.txt
  dataset/flickr8k/Flickr_8k.testImages.txt

Run:
    python src/notebook/captioning_preprocessing.py
"""

import re
import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

REPO_ROOT   = Path(__file__).resolve().parents[3]
DATA_DIR    = REPO_ROOT / 'dataset' / 'flickr8k'
PROC_DIR    = REPO_ROOT / 'data_processed'
PROC_DIR.mkdir(exist_ok=True)

IMG_DIR     = DATA_DIR / 'Images'
FEATURE_DIM = 2048      # InceptionV3 output
IMG_SIZE    = (299, 299)  # InceptionV3 input
MAX_LEN     = 35          # max caption length in tokens (excl. <start>/<end>)
MIN_FREQ    = 2           # minimum word frequency to keep in vocabulary

PAD_TOKEN   = '<pad>'
START_TOKEN = '<start>'
END_TOKEN   = '<end>'
UNK_TOKEN   = '<unk>'


def _clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_captions(data_dir: Path) -> dict:
    """Return {image_filename: [caption, ...]} dict."""
    csv_path = data_dir / 'captions.txt'
    if csv_path.exists():
        captions = {}
        with open(csv_path) as f:
            next(f)  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',', 1)
                if len(parts) < 2:
                    continue
                img, cap = parts[0].strip(), parts[1].strip()
                captions.setdefault(img, []).append(_clean(cap))
        return captions

    token_path = data_dir / 'Flickr8k.token.txt'
    if not token_path.exists():
        raise FileNotFoundError(
            f"No caption file found. Expected {csv_path} or {token_path}")
    captions = {}
    with open(token_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname_id, cap = line.split('\t', 1)
            fname = fname_id.split('#')[0]
            captions.setdefault(fname, []).append(_clean(cap))
    return captions


def load_split(data_dir: Path, split: str) -> list:
    """Return list of image filenames for the given split (train/val/test)."""
    name_map = {'train': 'Flickr_8k.trainImages.txt',
                'val':   'Flickr_8k.devImages.txt',
                'test':  'Flickr_8k.testImages.txt'}
    path = data_dir / name_map[split]
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]



def extract_features(img_dir: Path, img_names: list, batch_size: int = 32) -> dict:
    """Extract InceptionV3 features for a list of images.

    Returns {img_name: feature_vector (FEATURE_DIM,)}.
    """
    import tensorflow as tf
    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet', pooling='avg',
        input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False
    preprocess = tf.keras.applications.inception_v3.preprocess_input

    features = {}
    for start in range(0, len(img_names), batch_size):
        batch_names = img_names[start:start + batch_size]
        imgs = []
        valid_names = []
        for name in batch_names:
            path = img_dir / name
            if not path.exists():
                print(f'  [WARN] missing image: {path}')
                continue
            img = Image.open(path).convert('RGB').resize(IMG_SIZE)
            imgs.append(np.array(img, dtype=np.float32))
            valid_names.append(name)
        if not imgs:
            continue
        batch = preprocess(np.stack(imgs))
        feats = base_model.predict(batch, verbose=0)
        for name, feat in zip(valid_names, feats):
            features[name] = feat
        if (start // batch_size) % 10 == 0:
            print(f'  [{start + len(valid_names)}/{len(img_names)}] features extracted')
    return features



def build_vocabulary(captions: dict, train_imgs: list, min_freq: int = MIN_FREQ):
    """Build vocabulary from training captions only."""
    counter = Counter()
    for img in train_imgs:
        for cap in captions.get(img, []):
            counter.update(cap.split())

    vocab = {PAD_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2, UNK_TOKEN: 3}
    for word, freq in sorted(counter.items()):
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab



def encode_caption(caption: str, vocab: dict, max_len: int) -> tuple:
    """Encode caption to (decoder_input, decoder_target) integer arrays.

    decoder_input  (max_len,): [<start>, w₀, ..., w_{max_len-2}]  (may be padded)
    decoder_target (max_len,): [w₀, ..., w_{N-1}, <end>]          (may be padded)
    """
    words = caption.split()[:max_len - 1]  # reserve space for <end>
    token_ids = [vocab.get(w, vocab[UNK_TOKEN]) for w in words]

    dec_in  = [vocab[START_TOKEN]] + token_ids
    dec_in  = dec_in[:max_len]
    dec_in  += [vocab[PAD_TOKEN]] * (max_len - len(dec_in))

    dec_tgt = token_ids + [vocab[END_TOKEN]]
    dec_tgt = dec_tgt[:max_len]
    dec_tgt += [vocab[PAD_TOKEN]] * (max_len - len(dec_tgt))

    return np.array(dec_in, dtype=np.int32), np.array(dec_tgt, dtype=np.int32)


def prepare_dataset(features: dict, captions: dict, img_names: list,
                    vocab: dict, max_len: int):
    """Compile (img_feat, dec_in, dec_tgt) arrays for all images in img_names."""
    img_feats, dec_ins, dec_tgts = [], [], []
    for img in img_names:
        if img not in features:
            continue
        for cap in captions.get(img, []):
            dec_in, dec_tgt = encode_caption(cap, vocab, max_len)
            img_feats.append(features[img])
            dec_ins.append(dec_in)
            dec_tgts.append(dec_tgt)
    return (np.array(img_feats,  dtype=np.float32),
            np.array(dec_ins,    dtype=np.int32),
            np.array(dec_tgts,   dtype=np.int32))



def main():
    print('Loading captions …')
    captions = load_captions(DATA_DIR)
    print(f'  {len(captions)} unique images, '
          f'{sum(len(v) for v in captions.values())} total captions')

    try:
        train_imgs = load_split(DATA_DIR, 'train')
        val_imgs   = load_split(DATA_DIR, 'val')
        test_imgs  = load_split(DATA_DIR, 'test')
    except FileNotFoundError:
        print('  Split files not found — generating 6000/1000/1000 split …')
        all_imgs   = sorted(captions.keys())
        np.random.seed(42)
        np.random.shuffle(all_imgs)
        train_imgs = all_imgs[:6000]
        val_imgs   = all_imgs[6000:7000]
        test_imgs  = all_imgs[7000:]
        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            with open(DATA_DIR / f'Flickr_8k.{split}Images.txt', 'w') as f:
                f.write('\n'.join(imgs))

    print(f'  splits: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}')

    feat_path = PROC_DIR / 'image_features.pkl'
    if feat_path.exists():
        print('Loading cached image features …')
        with open(feat_path, 'rb') as f:
            features = pickle.load(f)
    else:
        print('Extracting InceptionV3 features …')
        all_imgs = list(set(train_imgs + val_imgs + test_imgs))
        features = extract_features(IMG_DIR, all_imgs)
        with open(feat_path, 'wb') as f:
            pickle.dump(features, f)
        print(f'  Saved features for {len(features)} images → {feat_path}')

    vocab_path = PROC_DIR / 'vocab.json'
    vocab = build_vocabulary(captions, train_imgs)
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f'Vocabulary size: {len(vocab)}  (saved to {vocab_path})')

    print('Encoding captions and preparing datasets …')
    for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        img_f, dec_in, dec_tgt = prepare_dataset(features, captions, imgs, vocab, MAX_LEN)
        np.save(PROC_DIR / f'{split}_img_feats.npy',   img_f)
        np.save(PROC_DIR / f'{split}_dec_input.npy',   dec_in)
        np.save(PROC_DIR / f'{split}_dec_target.npy',  dec_tgt)
        print(f'  {split}: {len(img_f)} samples  shapes: '
              f'img={img_f.shape}, dec_in={dec_in.shape}, dec_tgt={dec_tgt.shape}')

    ref_path = PROC_DIR / 'test_references.json'
    references = {img: captions[img] for img in test_imgs if img in captions}
    with open(ref_path, 'w') as f:
        json.dump(references, f, indent=2)
    print(f'Saved test references → {ref_path}')

    print('\nPreprocessing complete.')
    print(f'  feature_dim = {FEATURE_DIM}')
    print(f'  vocab_size  = {len(vocab)}')
    print(f'  max_len     = {MAX_LEN}')


if __name__ == '__main__':
    main()

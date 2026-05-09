# IF3270_Tubes2_61

Tugas Besar 2 IF3270 Machine Learning — Convolutional Neural Network & Image Captioning (RNN/LSTM).

## Dataset Setup

Dataset tidak disertakan di repo. Jalankan sekali sebelum training:

| Dataset | Dipakai untuk | Link |
|---|---|---|
| Intel Image Classification | CNN | https://www.kaggle.com/datasets/puneet6060/intel-image-classification |
| Flickr8k | RNN & LSTM Captioning | https://www.kaggle.com/datasets/adityajn105/flickr8k |


1. Install kagglehub (tidak perlu masuk venv project):
   ```
   pip install kagglehub
   ```

2. Siapkan Kaggle API token:
   - Buka https://www.kaggle.com/settings → bagian API
   - Klik "Create New API Token" → download `kaggle.json`
   - Taruh di `~/.kaggle/kaggle.json` (Linux/Mac)
             atau `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
   - Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

3. Jalankan download script:
   ```
   python dataset/download_datasets.py
   ```

Dataset akan tersimpan di `dataset/` (tidak di-commit ke git).

## Training

```bash
# CNN — Intel Image Classification (16 variants)
python src/notebook/training/cnn_training.py

# Captioning — preprocessing (run once)
python src/notebook/training/captioning_preprocessing.py

# Captioning — training (12 variants)
python src/notebook/training/captioning_training.py
```

## Evaluation

```bash
# CNN from-scratch vs Keras
python src/notebook/evaluation/cnn_evaluation.py

# Captioning from-scratch + BLEU-4 + METEOR
python src/notebook/evaluation/captioning_evaluation.py
```

## Notebook

Buka `Tubes2.ipynb` untuk analisis lengkap semua eksperimen.

## Requirements

```bash
pip install -r requirements.txt
pip install tensorflow scikit-learn nltk pillow matplotlib seaborn
```

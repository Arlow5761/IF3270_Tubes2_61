import shutil
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
DATASET_DIR.mkdir(exist_ok=True)

DATASETS = {
    'intel-image-classification': 'puneet6060/intel-image-classification',
    'flickr8k':                   'adityajn105/flickr8k',
}

try:
    import kagglehub
except ImportError:
    print('kagglehub tidak terinstall. Jalankan: pip install kagglehub')
    raise SystemExit(1)

for name, slug in DATASETS.items():
    dest = DATASET_DIR / name
    if dest.exists():
        print(f'[skip] {name} sudah ada di {dest}')
        continue
    print(f'Downloading {name} ({slug}) ...')
    cache_path = kagglehub.dataset_download(slug)
    shutil.copytree(cache_path, dest)
    print(f'  -> disimpan ke {dest}')

print(f'Dataset tersimpan di: {DATASET_DIR}')

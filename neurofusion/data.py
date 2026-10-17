import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['meningioma', 'glioma', 'pituitary']


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download():
    """Pinned Figshare v5; verifies the publisher's checksums before extraction."""
    source = json.loads((ROOT / 'data/source.json').read_text())
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
    for info in source['files']:
        path = raw / info['name']
        if not path.exists() or digest(path, 'md5') != info['computed_md5']:
            part = path.with_suffix(path.suffix + '.part')
            with session.get(info['download_url'], stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with open(part, 'wb') as f, tqdm(total=info['size'], unit='B', unit_scale=True, desc=path.name) as bar:
                    for chunk in response.iter_content(1024 * 1024):
                        f.write(chunk)
                        bar.update(len(chunk))
            if digest(part, 'md5') != info['computed_md5']:
                raise RuntimeError(f'Checksum mismatch: {part}. Retry download.')
            part.replace(path)
        if path.suffix == '.zip':
            destination = raw / 'mat'
            destination.mkdir(exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if not target.is_relative_to(destination.resolve()):
                        raise ValueError('Unsafe archive path')
                archive.extractall(destination)
    print('Download and publisher checksum verification complete.', flush=True)


def read_mat(path):
    with h5py.File(path) as f:
        obj = f['cjdata']
        label = int(np.asarray(obj['label']).item()) - 1
        patient = ''.join(chr(int(c)) for c in np.asarray(obj['PID']).ravel() if int(c))
        image = np.asarray(obj['image']).T  # MATLAB/HDF5 axis order
    if label not in range(3) or not patient or image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f'Invalid image, patient or label: {path}')
    return image, patient, label


def group_split(rows, seed=42):
    # Union patient groups connected by identical images. Duplicate images cannot cross splits.
    parent = {r['patient']: r['patient'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    patient_labels = {}
    for row in rows:
        pid, label, sha = row['patient'], row['label'], row['sha256']
        if pid in patient_labels and patient_labels[pid] != label:
            raise ValueError('Inconsistent patient labels')
        patient_labels[pid] = label
        if sha in first:
            other = first[sha]
            if other['label'] != label:
                raise ValueError('Identical pixels have conflicting labels')
            parent[find(pid)] = find(other['patient'])
        first[sha] = row
    groups = np.array([find(r['patient']) for r in rows])
    labels = np.array([r['label'] for r in rows])
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    development, test = next(outer.split(np.zeros(len(rows)), labels, groups))
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(np.zeros(len(development)), labels[development], groups[development]))
    splits = {'train': development[train_idx], 'val': development[val_idx], 'test': test}
    for split, indices in splits.items():
        if set(labels[indices]) != {0, 1, 2}:
            raise ValueError(f'{split} lacks a class; investigate data. Do not search seeds for test scores.')
        for i in indices:
            rows[i]['split'] = split
    return rows


def prepare():
    dest = ROOT / 'data/prepared'
    if (dest / 'manifest.csv').exists():
        raise RuntimeError('Manifest already exists. Keep the fixed split; do not regenerate during experiments.')
    images = dest / 'images'
    images.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / 'data/raw/mat').rglob('*.mat'))
    if len(files) != 3064:
        raise ValueError(f'Expected 3064 image files, found {len(files)}. Run download first.')
    rows = []
    for index, path in enumerate(tqdm(files, desc='Preparing MRI slices')):
        array, patient, label = read_mat(path)
        low, high = float(array.min()), float(array.max())
        if high <= low:
            raise ValueError(f'Constant image: {path}')
        # Per-image scaling uses no labels, annotations or statistics from other patients.
        normalized = np.rint(255 * (array.astype(np.float32) - low) / (high - low)).clip(0, 255).astype('uint8')
        filename = f'images/{index:04d}.png'
        Image.fromarray(normalized).save(dest / filename)
        rows.append(dict(path=filename, patient=patient, label=label,
                         sha256=hashlib.sha256(normalized.tobytes() + str(normalized.shape).encode()).hexdigest(),
                         file_sha256=digest(dest / filename), source_mat=path.relative_to(ROOT / 'data/raw/mat').as_posix()))
    rows = group_split(rows)
    with open(dest / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {s: dict(images=sum(r['split'] == s for r in rows),
                     patients=len({r['patient'] for r in rows if r['split'] == s}),
                     classes=dict(Counter(CLASSES[r['label']] for r in rows if r['split'] == s)))
             for s in ('train', 'val', 'test')}
    audit.update(seed=42, manifest_sha256=digest(dest / 'manifest.csv'),
                 exact_pixel_duplicate_rows=len(rows) - len({r['sha256'] for r in rows}),
                 note='Patient and exact-image disjoint; near-duplicate/site independence not established.')
    (dest / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


def load_rows(manifest):
    with open(manifest, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['label'] = int(r['label'])
    for key in ('patient', 'sha256'):
        seen = {}
        for r in rows:
            if r[key] in seen and seen[r[key]] != r['split']:
                raise ValueError(f'Cross-split leakage: {key}')
            seen[r[key]] = r['split']
    return rows


def verify_images(manifest):
    manifest = Path(manifest)
    rows = load_rows(manifest)
    for row in tqdm(rows, desc='Verify image checksums'):
        if digest(manifest.parent / row['path']) != row['file_sha256']:
            raise ValueError(f'Image changed since preparation: {row["path"]}')
    return rows


class MRIDataset(Dataset):
    def __init__(self, manifest, split, size=224):
        self.root = Path(manifest).parent
        self.rows = [r for r in load_rows(manifest) if r['split'] == split]
        if not self.rows:
            raise ValueError(f'Empty split: {split}')
        operations = [T.Resize((size, size))]
        if split == 'train':
            operations += [T.RandomAffine(10, translate=(0.03, 0.03), scale=(0.95, 1.05))]
        self.transform = T.Compose(operations + [T.ToTensor()])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, row['label'], row['patient'], row['path']

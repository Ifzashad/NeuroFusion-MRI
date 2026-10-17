import csv
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CLASSES, MRIDataset, ROOT, digest, verify_images
from .model import NeuroFusion


def seed_worker(worker):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def metrics(labels, probs):
    pred = np.asarray(probs).argmax(1)
    result = dict(accuracy=float(accuracy_score(labels, pred)),
                  macro_f1=float(f1_score(labels, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
                  confusion_matrix=confusion_matrix(labels, pred, labels=[0, 1, 2]).tolist(),
                  classification_report=classification_report(labels, pred, labels=[0, 1, 2],
                                                              target_names=CLASSES, output_dict=True, zero_division=0))
    result['macro_auc_ovr'] = float(roc_auc_score(labels, probs, labels=[0, 1, 2], multi_class='ovr')) if len(set(labels)) == 3 else None
    return result


def aggregate_patients(labels, probs, patients):
    grouped = {}
    for y, p, pid in zip(labels, probs, patients):
        if pid not in grouped:
            grouped[pid] = (int(y), [])
        if grouped[pid][0] != int(y):
            raise ValueError('Conflicting labels for patient')
        grouped[pid][1].append(p)
    return (np.array([v[0] for v in grouped.values()]),
            np.stack([np.mean(v[1], axis=0) for v in grouped.values()]), list(grouped))


def loader(manifest, split, cfg, device):
    return DataLoader(MRIDataset(manifest, split, cfg['image_size']), batch_size=cfg['batch_size'],
                      shuffle=split == 'train', num_workers=cfg['workers'], pin_memory=device.type == 'cuda',
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(cfg['seed']))


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    ys, ps, pids, paths = [], [], [], []
    for x, y, patient, path in tqdm(batches, desc='Evaluate', leave=False):
        probabilities = model(x.to(device)).softmax(1).cpu().numpy()
        ys.extend(y.tolist()); ps.extend(probabilities); pids.extend(patient); paths.extend(path)
    return np.array(ys), np.stack(ps), pids, paths


def make_device(value):
    if value == 'cpu':
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Fix GPU environment or explicitly select --device cpu.')
    device = torch.device(f'cuda:{int(value)}')
    torch.cuda.set_device(device)
    print('Device:', device, torch.cuda.get_device_name(device), 'CUDA_VISIBLE_DEVICES:', os.getenv('CUDA_VISIBLE_DEVICES', '<unset>'), flush=True)
    return device


def train(config, output, device_value, scratch=False):
    cfg = yaml.safe_load(Path(config).read_text())
    device = make_device(device_value)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    manifest = ROOT / 'data/prepared/manifest.csv'
    verify_images(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never silently overwrite a run.
    training, validation = loader(manifest, 'train', cfg, device), loader(manifest, 'val', cfg, device)
    model = NeuroFusion(cfg['variant'], pretrained=not scratch, channels=cfg['channels'], dropout=cfg['dropout']).to(device)
    backbone_params = list(model.backbones.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([dict(params=backbone_params, lr=cfg['backbone_lr']),
                                   dict(params=head_params, lr=cfg['head_lr'])], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    counts = np.bincount([r['label'] for r in training.dataset.rows], minlength=3)
    if min(counts) == 0:
        raise ValueError('Training split missing a class')
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg['label_smoothing'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    cfg.update(pretrained=not scratch, classes=CLASSES, manifest_sha256=digest(manifest))
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg))
    (output / 'environment.txt').write_text(platform.platform() + '\n' + subprocess.check_output(
        [os.sys.executable, '-m', 'pip', 'freeze'], text=True))
    best, stale, started = -1.0, 0, time.time()
    for epoch in range(cfg['epochs']):
        model.freeze_backbones(epoch < cfg['freeze_epochs'])
        model.train()
        total_loss, count = 0.0, 0
        for x, y, _, _ in tqdm(training, desc=f'Epoch {epoch + 1}/{cfg["epochs"]}'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; stopping without publishing results')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * len(y); count += len(y)
        labels, probs, patients, _ = predict(model, validation, device)
        py, pp, _ = aggregate_patients(labels, probs, patients)
        score = metrics(py, pp)['macro_f1']
        record = dict(epoch=epoch + 1, train_loss=total_loss / count,
                      val_slice_accuracy=metrics(labels, probs)['accuracy'], val_patient_macro_f1=score,
                      elapsed_seconds=time.time() - started)
        with open(output / 'history.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            if epoch == 0:
                writer.writeheader()
            writer.writerow(record)
        improved = score > best + cfg['min_delta']
        if improved:
            best, stale = score, 0
            torch.save(dict(model=model.state_dict(), config=cfg, epoch=epoch + 1,
                            validation_patient_macro_f1=score), output / 'best.pt')
        elif epoch >= cfg['freeze_epochs']:
            stale += 1
        scheduler.step()
        torch.save(dict(model=model.state_dict(), config=cfg, epoch=epoch + 1), output / 'last.pt')
        print(json.dumps(record), 'best:', best, 'patience:', stale, flush=True)
        if stale >= cfg['patience']:
            print('Early stopping: validation patient macro-F1 stopped improving.', flush=True)
            break
    (output / 'complete.json').write_text(json.dumps(dict(best_validation_patient_macro_f1=best,
                                                       epochs_completed=epoch + 1, seconds=time.time() - started), indent=2))
    print('TRAINING COMPLETE:', output, 'Test set has not been evaluated.', flush=True)


def evaluate(weights, output, device_value, split='test'):
    device = make_device(device_value)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    manifest = ROOT / 'data/prepared/manifest.csv'
    if digest(manifest) != cfg['manifest_sha256']:
        raise ValueError('Manifest differs from the training split')
    verify_images(manifest)
    model = NeuroFusion(cfg['variant'], pretrained=False, channels=cfg['channels'], dropout=cfg['dropout']).to(device)
    model.load_state_dict(checkpoint['model'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    y, p, ids, paths = predict(model, loader(manifest, split, cfg, device), device)
    py, pp, patients = aggregate_patients(y, p, ids)
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(1000):
        # Stratified patient bootstrap: condition on the observed class composition.
        indices = np.concatenate([rng.choice(np.flatnonzero(py == c), size=sum(py == c), replace=True) for c in range(3)])
        scores.append(f1_score(py[indices], pp[indices].argmax(1), labels=[0, 1, 2], average='macro', zero_division=0))
    result = dict(split=split, variant=cfg['variant'], images=len(y), patients=len(py),
                  slice=metrics(y, p), patient=metrics(py, pp),
                  patient_macro_f1_stratified_bootstrap_ci95=np.percentile(scores, [2.5, 97.5]).tolist(),
                  checkpoint_sha256=digest(weights), manifest_sha256=digest(manifest),
                  checkpoint_epoch=checkpoint['epoch'])
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))
    for name, labels, probs, keys in [('slice', y, p, paths), ('patient', py, pp, patients)]:
        with open(output / f'{name}-predictions.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'label', 'prediction'] + [f'p_{c}' for c in CLASSES])
            writer.writerows([key, int(label), int(prob.argmax()), *map(float, prob)] for key, label, prob in zip(keys, labels, probs))
    make_plots(py, pp, output)
    print(json.dumps(result, indent=2), flush=True)


def make_plots(labels, probs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(labels, probs.argmax(1), labels=[0, 1, 2], display_labels=CLASSES, ax=ax, colorbar=False)
    ax.set_title('Patient-level confusion matrix')
    fig.tight_layout(); fig.savefig(output / 'confusion-matrix.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, name in enumerate(CLASSES):
        truth = np.asarray(labels) == c
        if len(set(truth)) == 2:
            fpr, tpr, _ = roc_curve(truth, probs[:, c])
            ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Patient-level one-vs-rest ROC')
    ax.legend(); fig.tight_layout(); fig.savefig(output / 'roc.png', dpi=160); plt.close(fig)

"""Exercise checkpoint selection and held-out evaluation on synthetic images."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import yaml

from neurofusion import engine
from neurofusion.data import digest


class TinyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbones = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(8, 3)

    def freeze_backbones(self, frozen):
        self.backbones.requires_grad_(not frozen)

    def forward(self, x):
        return self.head(self.backbones(x).mean((2, 3)))


class EngineTests(unittest.TestCase):
    def test_training_checkpoint_evaluation_and_split_guard(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / 'data/prepared'
            dest.mkdir(parents=True)
            rows = []
            rng = np.random.default_rng(7)
            for split in ('train', 'val', 'test'):
                for c in range(3):
                    path = dest / f'{split}-{c}.png'
                    Image.fromarray(rng.integers(0, 255, (24, 24), dtype=np.uint8)).save(path)
                    rows.append(dict(path=path.name, patient=path.stem, label=c, split=split, sha256=digest(path), file_sha256=digest(path)))
            manifest = dest / 'manifest.csv'
            with open(manifest, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            cfg = dict(variant='full',seed=42,image_size=24,batch_size=3,workers=0,channels=16,
                       dropout=.4,epochs=2,freeze_epochs=1,patience=1,min_delta=.0001,
                       backbone_lr=.001,head_lr=.001,weight_decay=.0001,label_smoothing=.05)
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump(cfg))
            with patch.object(engine, 'ROOT', root), patch.object(engine, 'NeuroFusion', TinyModel):
                engine.train(config, root / 'run', 'cpu', scratch=True)
                self.assertTrue((root / 'run/complete.json').exists())
                engine.evaluate(root / 'run/best.pt', root / 'test', 'cpu')
                result = json.loads((root / 'test/metrics.json').read_text())
                self.assertEqual(result['images'], 3)
                self.assertEqual(result['patients'], 3)
                self.assertEqual(result['checkpoint_sha256'], digest(root / 'run/best.pt'))
                self.assertTrue((root / 'test/confusion-matrix.png').exists())
                with open(manifest, 'a') as f:
                    f.write('\n')
                with self.assertRaisesRegex(ValueError, 'Manifest differs'):
                    engine.evaluate(root / 'run/best.pt', root / 'bad-eval', 'cpu')

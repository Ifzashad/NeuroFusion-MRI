import copy
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from neurofusion.data import group_split, read_mat
from neurofusion.engine import aggregate_patients, metrics
from neurofusion.model import NeuroFusion, VARIANTS


class PipelineTests(unittest.TestCase):
    def test_patient_and_duplicate_separation(self):
        rows = [dict(patient=f'p{c}-{p}', label=c, sha256=f'{c}-{p}-{s}')
                for c in range(3) for p in range(20) for s in range(2)]
        rows[2]['sha256'] = rows[0]['sha256']
        a, b = group_split(copy.deepcopy(rows)), group_split(copy.deepcopy(rows))
        self.assertEqual(a, b)
        for field in ('patient', 'sha256'):
            seen = {}
            for row in a:
                self.assertIn(row['split'], ('train', 'val', 'test'))
                self.assertEqual(seen.setdefault(row[field], row['split']), row['split'])
        self.assertEqual(a[0]['split'], a[2]['split'])

    def test_conflicting_duplicates_rejected(self):
        rows = [dict(patient='a', label=0, sha256='same'), dict(patient='b', label=1, sha256='same')]
        with self.assertRaises(ValueError):
            group_split(rows)

    def test_mat_orientation_label_and_patient(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'image.mat'
            image = np.arange(20).reshape(4, 5)
            with h5py.File(path, 'w') as f:
                group = f.create_group('cjdata')
                group['image'] = image.T
                group['label'] = [[2]]
                group['PID'] = np.array([ord(x) for x in 'patient-17'])[:, None]
            actual, patient, label = read_mat(path)
            np.testing.assert_array_equal(actual, image)
            self.assertEqual((patient, label), ('patient-17', 1))

    def test_patient_probability_aggregation(self):
        labels = [0, 0, 1, 2]
        probs = np.array([[.8, .1, .1], [.6, .3, .1], [.1, .8, .1], [.1, .1, .8]])
        y, p, ids = aggregate_patients(labels, probs, ['a', 'a', 'b', 'c'])
        np.testing.assert_allclose(p[0], [.7, .2, .1])
        self.assertEqual(metrics(y, p)['accuracy'], 1.0)
        self.assertEqual(len(ids), 3)

    def test_real_backbones_gradients_and_checkpoint(self):
        torch.set_num_threads(2)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = NeuroFusion(variant, pretrained=False, channels=16)
                model.freeze_backbones(True)
                model.train()
                self.assertFalse(model.backbones[0].training)
                x = torch.rand(2, 3, 96, 96)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, 3))
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
                self.assertTrue(torch.isfinite(model.head[-1].weight.grad).all())
                if model.recurrent is not None:
                    self.assertGreater(model.recurrent.gates.weight.grad.abs().sum().item(), 0)
                self.assertIsNone(next(model.backbones.parameters()).grad)
                model.freeze_backbones(False)
                self.assertTrue(next(model.backbones.parameters()).requires_grad)
                model.eval()
                with torch.no_grad():
                    before = model(x)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'model.pt'
                    torch.save(model.state_dict(), path)
                    model.load_state_dict(torch.load(path, weights_only=True))
                with torch.no_grad():
                    torch.testing.assert_close(before, model(x))
                if variant == 'full':
                    model.train()
                    model.zero_grad(set_to_none=True)
                    model(x).sum().backward()
                    for backbone in model.backbones:
                        gradient = next(backbone.parameters()).grad
                        self.assertIsNotNone(gradient)
                        self.assertTrue(torch.isfinite(gradient).all())

    def test_gradcam_output(self):
        from PIL import Image
        from neurofusion.explain import explain
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = NeuroFusion('full', pretrained=False, channels=16)
            checkpoint = root / 'best.pt'
            torch.save(dict(model=model.state_dict(), config=dict(variant='full', channels=16, dropout=.4, image_size=96)), checkpoint)
            image = root / 'slice.png'
            Image.fromarray(np.random.default_rng(1).integers(0, 255, (96, 96), dtype=np.uint8)).save(image)
            explain(checkpoint, image, root / 'explanation', 'cpu')
            with Image.open(root / 'explanation/gradcam.png') as actual:
                self.assertEqual(actual.size, (96, 96))


if __name__ == '__main__':
    unittest.main()

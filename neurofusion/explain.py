import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor

from .data import CLASSES, digest
from .engine import make_device
from .model import NeuroFusion


def explain(weights, image_path, output, device_value):
    """Grad-CAM on the final fused spatial feature map; no ground-truth label input."""
    device = make_device(device_value)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    model = NeuroFusion(cfg['variant'], pretrained=False, channels=cfg['channels'], dropout=cfg['dropout']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert('L'), dtype=np.float32)
    if gray.max() <= gray.min():
        raise ValueError('Constant image')
    gray = np.rint(255 * (gray - gray.min()) / (gray.max() - gray.min())).astype(np.uint8)
    image = Image.fromarray(gray).convert('RGB').resize((cfg['image_size'], cfg['image_size']), Image.Resampling.BILINEAR)
    x = to_tensor(image).unsqueeze(0).to(device)
    features = model.forward_features(x)
    features.retain_grad()
    logits = model.head(features)
    prediction = int(logits.argmax(1).item())
    logits[0, prediction].backward()
    cam = (features.grad.mean((2, 3), keepdim=True) * features).sum(1, keepdim=True).relu()
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-8)
    import matplotlib
    colors = matplotlib.colormaps['inferno'](cam)[..., :3]
    overlay = np.clip(.6 * np.asarray(image) + .4 * colors * 255, 0, 255).astype(np.uint8)
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    Image.fromarray(overlay).save(dest / 'gradcam.png')
    probabilities = logits.softmax(1)[0].detach().cpu().tolist()
    result = dict(prediction=CLASSES[prediction], probabilities=dict(zip(CLASSES, probabilities)),
                  checkpoint_sha256=digest(weights),
                  note='Research-only closed-set classification. No healthy class. Grad-CAM is not a segmentation or proof of clinical reasoning.')
    (dest / 'prediction.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

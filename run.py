"""Portable command-line entry point. All caches stay inside this project."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    p = argparse.ArgumentParser()
    commands = p.add_subparsers(dest='command', required=True)
    commands.add_parser('download')
    commands.add_parser('prepare')
    train = commands.add_parser('train')
    train.add_argument('--config', default=str(ROOT / 'configs/full.yaml'))
    train.add_argument('--output', required=True)
    train.add_argument('--device', required=True, help='Visible CUDA index or cpu')
    train.add_argument('--scratch', action='store_true', help='No pretrained download; only for pipeline checks')
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--weights', required=True)
    evaluation.add_argument('--output', required=True)
    evaluation.add_argument('--device', required=True)
    evaluation.add_argument('--split', choices=['val', 'test'], default='test')
    explanation = commands.add_parser('explain')
    explanation.add_argument('--weights', required=True)
    explanation.add_argument('--image', required=True)
    explanation.add_argument('--output', required=True)
    explanation.add_argument('--device', required=True)
    args = p.parse_args()
    if args.command in ('download', 'prepare'):
        from neurofusion import data
        getattr(data, args.command)()
    elif args.command == 'train':
        from neurofusion.engine import train
        train(args.config, args.output, args.device, args.scratch)
    elif args.command == 'evaluate':
        from neurofusion.engine import evaluate
        evaluate(args.weights, args.output, args.device, args.split)
    else:
        from neurofusion.explain import explain
        explain(args.weights, args.image, args.output, args.device)


if __name__ == '__main__':
    main()

import argparse
import glob
import json
import os

import torch
from torch.utils.data import DataLoader

from utils import (OnsetCounter, WindowDataset, build_model, get_device, list_files, peaks, predict,
                   sweep_thetas)


def score(model, files, theta, device, bs):
    # onset f1 with 1 frame (32 ms) and 2 frames (64 ms) tolerance
    loader = DataLoader(WindowDataset(files), batch_size=bs, num_workers=2)
    counters = {1: OnsetCounter(1), 2: OnsetCounter(2)}
    for p_onset, label in predict(model, loader, device):
        onsets = peaks(p_onset, theta)
        for c in counters.values():
            c.update(onsets, label)
    return {f'tol{t}': c.result() for t, c in counters.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--extra')                       # another folder with .pt files, e.g. smd
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=8)
    args = ap.parse_args()

    device = get_device()
    model = build_model(args.mode).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))

    # threshold is picked on validation, the test split is only used once
    val_loader = DataLoader(WindowDataset(list_files(args.data, 'validation'), per_file=4),
                            batch_size=args.bs, num_workers=2)
    val = sweep_thetas(model, val_loader, device)
    theta = max(val, key=lambda t: val[t]['f1'])
    print('validation f1:', {t: round(v['f1'], 4) for t, v in val.items()}, 'theta', theta)

    result = {'mode': args.mode, 'theta': theta, 'test': score(model, list_files(args.data, 'test'), theta, device, args.bs)}
    if args.extra:
        result['extra'] = score(model, sorted(glob.glob(os.path.join(args.extra, '*.pt'))), theta, device, args.bs)
    print(json.dumps(result['test'], indent=1))
    json.dump(result, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import (CropDataset, WindowDataset, build_model, get_device, list_files, n_params,
                   sweep_thetas)


def teacher_forcing(epoch, epochs, hold, floor):
    # 1 for the first epochs, then linear down to floor
    if epoch <= hold:
        return 1.0
    return max(floor, 1.0 - (epoch - hold) / max(1, epochs - hold) * (1.0 - floor))


def kd_loss(s_logits, t_logits, labels, weights, T, alpha, use_mask):
    s = s_logits.reshape(-1, 5)
    t = t_logits.reshape(-1, 5)
    y = labels.reshape(-1)

    hard = F.cross_entropy(s, y, weight=weights)
    kl = F.kl_div(F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction='none').sum(-1)
    if use_mask:
        # only the cells where there is an onset or the teacher thinks there could be one
        mask = ((y == 3) | (F.softmax(t, -1)[:, 3] > 0.05)).float()
        soft = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        soft = kl.mean()
    return alpha * hard + (1 - alpha) * soft * T ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True)          # teacher, student, student_pool4, ...
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--teacher')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weights', type=float, nargs=4, default=[1.0, 3.0, 1.5, 5.0])
    ap.add_argument('--alpha', type=float, default=0.93)
    ap.add_argument('--temp', type=float, default=3.3)
    ap.add_argument('--no_mask', action='store_true')
    ap.add_argument('--no_kd', action='store_true')
    ap.add_argument('--tf_hold', type=int, default=6)
    ap.add_argument('--tf_floor', type=float, default=0.25)
    ap.add_argument('--seq_len', type=int, default=1024)
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--extra_max', type=int, default=150)
    ap.add_argument('--val_files', type=int, default=60)
    ap.add_argument('--val_every', type=int, default=2)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = get_device()

    train_files = list_files(args.data, 'train')
    extra = []
    for d in args.extra:
        extra += sorted(glob.glob(os.path.join(d, '*.pt')))
    if len(extra) > args.extra_max:
        extra = random.Random(args.seed).sample(extra, args.extra_max)
    train_files += extra
    val_files = list_files(args.data, 'validation')
    print(f'{len(train_files)} train files ({len(extra)} synthetic), {len(val_files)} validation files')

    model = build_model(args.mode).to(device)
    print(f'{args.mode}: {n_params(model):,} parameters')

    teacher = None
    if args.mode != 'teacher' and not args.no_kd:
        teacher = build_model('teacher').to(device)
        teacher.load_state_dict(torch.load(args.teacher, map_location='cpu'))
        teacher.eval()

    weights = torch.tensor(args.weights + [0.0], device=device)    # 5th class is never used
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)

    train_loader = DataLoader(CropDataset(train_files, args.seq_len), batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_set = WindowDataset(val_files, args.seq_len, per_file=2, max_files=args.val_files)
    val_loader = DataLoader(val_set, batch_size=8, num_workers=args.workers)

    start, best, bad = 1, -1.0, 0
    last_path = os.path.join(args.out, 'last.pt')
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location='cpu')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start, best, bad = ck['epoch'] + 1, ck['best'], ck['bad']
        print('resumed from epoch', ck['epoch'])

    for epoch in range(start, args.epochs + 1):
        model.train()
        tf = teacher_forcing(epoch, args.epochs, args.tf_hold, args.tf_floor)
        losses = []
        for step, (mel, label) in enumerate(train_loader):
            mel, label = mel.to(device), label.to(device)
            # teacher forcing is decided for the whole batch, so the fast path can be used
            use_gt = 1.0 if random.random() < tf else 0.0
            logits = model(mel, gt_label=label, teacher_forcing_ratio=use_gt)

            if teacher is None:
                loss = F.cross_entropy(logits.reshape(-1, 5), label.reshape(-1), weight=weights)
            else:
                with torch.no_grad():
                    t_logits = teacher(mel, gt_label=label, teacher_forcing_ratio=1.0)
                loss = kd_loss(logits, t_logits, label, weights, args.temp, args.alpha, not args.no_mask)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f'epoch {epoch} step {step} loss {losses[-1]:.4f}')
        sched.step()

        record = {'epoch': epoch, 'loss': float(np.mean(losses)), 'tf': tf}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            res = sweep_thetas(model, val_loader, device)
            theta = max(res, key=lambda t: res[t]['f1'])
            record.update(val_f1=res[theta]['f1'], theta=theta)
            if record['val_f1'] > best:
                best, bad = record['val_f1'], 0
                torch.save(model.state_dict(), os.path.join(args.out, 'best.pth'))
            else:
                bad += 1
        print(json.dumps(record))
        with open(os.path.join(args.out, 'log.jsonl'), 'a') as f:
            f.write(json.dumps(record) + '\n')

        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': epoch, 'best': best, 'bad': bad}, last_path)
        if bad >= args.patience:
            print('no improvement on validation, stopping')
            break


if __name__ == '__main__':
    main()

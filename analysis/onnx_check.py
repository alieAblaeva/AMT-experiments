import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import build_model


class Acoustic(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model.acoustic_model

    def forward(self, mel):                         # (1, T, 229) -> (1, T, 256)
        return self.net(mel)


class GruStep(nn.Module):
    # one frame: features + embedding of the previous prediction + hidden state
    def __init__(self, model):
        super().__init__()
        self.gru = model.language_model
        self.post = model.language_post

    def forward(self, feat, prev_emb, h):
        out, h = self.gru(torch.cat([feat, prev_emb], 2), h)
        return self.post(torch.cat([out, feat], 2)).reshape(1, 88, 5), h


def run(acoustic, step, mel, emb):
    feat = acoustic.run(None, {'mel': mel[None]})[0]
    h = np.zeros((1, 1, 384), np.float32)
    prev = np.zeros((1, 1, 176), np.float32)
    logits = []
    for t in range(feat.shape[1]):
        out, h = step.run(None, {'feat': feat[:, t:t + 1], 'prev_emb': prev, 'h': h})
        prev = emb[out.argmax(-1)].reshape(1, 1, 176).astype(np.float32)
        logits.append(out[0])
    return np.stack(logits)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--windows', type=int, default=20)
    args = ap.parse_args()

    model = build_model('student')
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.eval()
    emb = model.class_embedding.weight.detach().numpy()

    os.makedirs('onnx', exist_ok=True)
    torch.onnx.export(Acoustic(model).eval(), torch.randn(1, 64, 229), 'onnx/acoustic.onnx', input_names=['mel'],
                      output_names=['feat'], dynamic_axes={'mel': {1: 'T'}, 'feat': {1: 'T'}}, opset_version=17, dynamo=False)
    torch.onnx.export(GruStep(model).eval(), (torch.randn(1, 1, 256), torch.zeros(1, 1, 176), torch.zeros(1, 1, 384)),
                      'onnx/gru_step.onnx', input_names=['feat', 'prev_emb', 'h'], output_names=['logits', 'h_out'],
                      opset_version=17, dynamo=False)
    for name in ('acoustic', 'gru_step'):
        quantize_dynamic(f'onnx/{name}.onnx', f'onnx/{name}_int8.onnx', weight_type=QuantType.QInt8)

    sizes = {n: round(os.path.getsize(f'onnx/{n}.onnx') / 1e6, 2) for n in
             ('acoustic', 'gru_step', 'acoustic_int8', 'gru_step_int8')}
    print('size, MB:', sizes)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sessions = {k: [ort.InferenceSession(f'onnx/{n}{k}.onnx', opts) for n in ('acoustic', 'gru_step')] for k in ('', '_int8')}

    files = sorted(glob.glob(os.path.join(args.cache, '*.pt')))
    same = {'': [], '_int8': []}
    for f in files:
        data = torch.load(f)
        for q in (0.25, 0.6):
            start = int(data['mel'].shape[0] * q)
            mel = data['mel'][start:start + 512].float()
            if len(same['']) >= args.windows:
                break
            with torch.no_grad():
                ref = model(mel[None]).numpy()[0]
            for k, (a, g) in sessions.items():
                out = run(a, g, mel.numpy(), emb)
                same[k].append((out.argmax(-1) == ref.argmax(-1)).mean())
    print('same decisions as pytorch:', {k or 'fp32': round(float(np.mean(v)), 4) for k, v in same.items()})

    for k, (a, g) in sessions.items():
        x = {'mel': np.random.randn(1, 5, 229).astype(np.float32)}
        y = {'feat': np.random.randn(1, 1, 256).astype(np.float32), 'prev_emb': np.zeros((1, 1, 176), np.float32),
             'h': np.zeros((1, 1, 384), np.float32)}
        t0 = time.perf_counter()
        for _ in range(200):
            a.run(None, x)
        t_a = (time.perf_counter() - t0) / 200 * 1000
        t0 = time.perf_counter()
        for _ in range(200):
            g.run(None, y)
        t_g = (time.perf_counter() - t0) / 200 * 1000
        print(f"{k or 'fp32'}: {t_a / 5 + t_g:.2f} ms per 32 ms frame")

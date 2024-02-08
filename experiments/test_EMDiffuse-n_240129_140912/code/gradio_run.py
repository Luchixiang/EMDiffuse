from __future__ import annotations

import functools
import pickle
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

sys.path.insert(0, 'StyleGAN-Human')

TITLE = 'StyleGAN-Human'
DESCRIPTION = 'https://github.com/stylegan-human/StyleGAN-Human'


def generate_z(z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(
        1, z_dim)).to(device).float()


@torch.inference_mode()
def generate_image(seed: int, truncation_psi: float, model: nn.Module,
                   device: torch.device) -> np.ndarray:
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))

    z = generate_z(model.z_dim, seed, device)
    label = torch.zeros([1, model.c_dim], device=device)

    out = model(z, label, truncation_psi=truncation_psi, force_fp32=True)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return out[0].cpu().numpy()


def load_model(file_name: str, device: torch.device) -> nn.Module:
    path = hf_hub_download('public-data/StyleGAN-Human', f'models/{file_name}')
    with open(path, 'rb') as f:
        model = pickle.load(f)['G_ema']
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.z_dim)).to(device)
        label = torch.zeros([1, model.c_dim], device=device)
        model(z, label, force_fp32=True)
    return model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model('stylegan_human_v2_1024.pkl', device)
fn = functools.partial(generate_image, model=model, device=device)

gr.Interface(
    fn=fn,
    inputs=[
        gr.Slider(label='Seed', minimum=0, maximum=100000, step=1, value=0),
        gr.Slider(label='Truncation psi',
                  minimum=0,
                  maximum=2,
                  step=0.05,
                  value=0.7),
    ],
    outputs=gr.Image(label='Output', type='numpy'),
    title=TITLE,
    description=DESCRIPTION,
).queue(max_size=10).launch()

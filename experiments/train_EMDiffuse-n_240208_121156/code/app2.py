import argparse
import os
import warnings
import torch
from torchvision import transforms
import gradio as gr
import PIL
import json
from tqdm import tqdm

means = [0.5]
stds = [0.5]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
t_stds = torch.tensor(stds).to(device)[:, None, None]
t_means = torch.tensor(means).to(device)[:, None, None]


def tensor2im(var):
    return var.mul(t_stds).add(t_means).mul(255.).clamp(0, 255)


def process(input_image, step=None):
    progressbar = gr.Progress(track_tqdm=True)
    img = tfs(input_image)[None, ...].to(device)
    b, *_ = img.shape
    if step == None:
        timesteps = 200
    else:
        timesteps = step
    with torch.no_grad():
        model.eval()
        y_t = torch.randn_like(img)
        count = timesteps
        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            t = torch.full((b,), count, device=img.device, dtype=torch.long)
            y_t = model(img, y_t, t)
            count -= 1
            # out, _ = model.netG.restoration(img, sample_num=8)
        out = y_t
        out = out[0, :, :, :]
        out = tensor2im(out)[0, :, :]
        output_image = out.detach().cpu().numpy().astype('uint8')
        output_image = PIL.Image.fromarray(output_image)
    return output_image


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False

    model = torch.jit.load('./emdiffuse-n.pt', map_location=torch.device('cpu')).to(device)
    tfs = transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    title = "EMDiffuse-n"
    description = "EMDiffuse-n demo for electron micrscope image denoising. \n" \
                  "The model is trained on mouse brain cortex scanning electron microscope dataset. To use it, simply upload your image, or click one of the examples to load them.\n" \
                  "Gradio does not currently support the TIFF image format, so the uploaded image may not appear in the input box, but it still functions when you click submit. Read more at the links below."
    article = "<div style='text-align: center;'>Chixiang et al. <a href='https://www.biorxiv.org/content/10.1101/2023.07.12.548636v1' target='_blank'> EMDiffuse S</a> | <a href='https://github.com/Luchixiang/EMDiffuse' target='_blank'>Github Repo</a> | <center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_arcanegan' alt='visitor badge'></center></div>"
    gr.Interface(
        process,
        inputs=[gr.Image(type="pil", label="Input", image_mode='L'), gr.Slider(100, 1000, label='Step numbers',
                                                                               info="The diffusion model's step count directly impacts image quality: more steps lead to higher quality images.")],
        outputs=[gr.Image(type="pil", label="Output", image_mode='L')],
        title=title,
        description=description,
        article=article,
        examples=[['./demo/22.png', None], ['./demo/23.png', None], ['./demo/24.png', None]],
        allow_flagging=False,
    ).queue(max_size=10).launch()

import argparse
import os
import warnings
import torch

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
from torchvision import transforms
import gradio as gr
import PIL

torch.cuda.set_per_process_memory_fraction(0.65, 0)

def model_creation(gpu, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    '''set seed and and cuDNN environment '''
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.enabled = False
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt=opt,
        networks=networks,
        phase_loader=None,
        val_loader=None,
        losses=losses,
        metrics=metrics,
        logger=phase_logger,
        writer=phase_writer
    )

    return model

means = [0.5]
stds = [0.5]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
t_stds = torch.tensor(stds).to(device)[:, None, None]
t_means = torch.tensor(means).to(device)[:, None, None]


def tensor2im(var):
    return var.mul(t_stds).add(t_means).mul(255.).clamp(0, 255)


def process(step, input_image, progress=gr.Progress()):
    img = tfs(input_image)[None, ...].to(device)
    timesteps =step
    with torch.no_grad():
        model.netG.eval()
        b, *_ = img.shape
        y_t = torch.randn_like(img)
        for i in progress.tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            t = torch.full((b,), i, device=img.device, dtype=torch.long)
            y_t, y_0_hat = model.netG.p_sample(y_t, t, y_cond=img, path=None, adjust=False)
            # out, _ = model.netG.restoration(img, sample_num=8)
        out = y_t
        out = out[0, :, :, :]
        out = tensor2im(out)[0, :, :]
        output_image = out.detach().cpu().numpy().astype('uint8')
        output_image = PIL.Image.fromarray(output_image)
    return output_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='patch of cropped patches')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-mean', type=int, default=1)
    parser.add_argument('-d', '--debug', action='store_true')
    # parser.add_argument('-')

    ''' parser configs '''
    args = parser.parse_args()
    args.config = './config/EMDiffuse-n-big.json'
    args.phase = 'test'
    args.batch = 1
    args.z_times = None

    opt = Praser.parse(args)
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

    opt['world_size'] = 1

    model = model_creation(0, opt)
    tfs = transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    title = "EMDiffuse-n"
    description = "Gradio demo for EMDiffuse-n for electron micrscope image denoising. \n" \
                  "The model is trained on mouse brain cortex scanning electron microscope dataset. To use it, simply upload your image, or click one of the examples to load them.\n" \
                  "Gradio does not currently support the TIFF image format, so it may not appear in the input box, but it still functions when you click submit. Read more at the links below."
    article = "<div style='text-align: center;'>ArcaneGan by <a href='https://twitter.com/devdef' target='_blank'>Alexander S</a> | <a href='https://github.com/Sxela/ArcaneGAN' target='_blank'>Github Repo</a> | <center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_arcanegan' alt='visitor badge'></center></div>"
    gr.Interface(
        process,
        inputs=[gr.Slider(200, 1000,label='Step numbers', info="The diffusion model's step count directly impacts image quality: more steps lead to higher quality images."), gr.Image(type="pil", label="Input", image_mode='L')],
        outputs=[gr.Image(type="pil", label="Output", image_mode='L')],
        title=title,
        description=description,
        article=article,
        examples=[[None, './demo/22.png'], [None, './demo/23.png'], [None, './demo/24.png']],
        allow_flagging=False,
    ).queue(max_size=10).launch()

    # with

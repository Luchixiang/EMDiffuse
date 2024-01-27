import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm
from bioimageio.core.build_spec import build_model, add_weights



def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt['init_method'],
                                             world_size=opt['world_size'],
                                             rank=opt['global_rank'],
                                             group_name='mtorch'
                                             )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt)  # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt=opt,
        networks=networks,
        phase_loader=phase_loader,
        val_loader=val_loader,
        losses=losses,
        metrics=metrics,
        logger=phase_logger,
        writer=phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    network = model.netG
    network = torch.jit.script(network)

    os.makedirs('bioimage_zoo_model', exist_ok=True)
    network.save("bioimage_zoo_model/bioimagezoo_weights_emdiffuse-n.pt")
    # input_ = np.random.rand(1, 1, 768, 768).astype("float32")  # an example input
    input_ = imread('./demo/27.tif')

    input_ = np.expand_dims(input_, axis=0)
    input_ = np.expand_dims(input_, axis=0).astype(np.float32)
    np.save("bioimage_zoo_model/test-input.npy", input_)
    input_ = ((input_ / 255) - 0.5) / 0.5
    num_timesteps = 1000
    y_t = torch.rand(input_.shape).cuda()
    with torch.no_grad():
        for i in tqdm(reversed(range(0, num_timesteps)), desc='sampling loop time step', total=num_timesteps):
            y_t = network(torch.from_numpy(input_).cuda(), y_t, i)
        output = y_t.cpu().numpy()
    output  = ((output + 1) * 127.5).round()
    output = output.astype(np.uint8)
    print(output.shape)
    # imwrite('./9_out.tif', output)
    np.save("bioimage_zoo_model/test-output.npy", output)
    with open("bioimage_zoo_model/doc.md", "w") as f:
        f.write("# EMDiffuse: a diffusion-based deep learning method augmenting ultrastructural imaging and volume electron microscopy. \n")
        f.write("EMDiffuse is a package for the application of diffusion models on electron microscopy images, aiming to enhance EM ultrastructural imaging and expand the realm of vEM capabilities. Here, we adopted the diffusion model for EM applications  \n")
        f.write("This model is EMDiffuse-n trained on mouse brain cortex dataset.\n")
    build_model(
        # the weight file and the type of the weights
        weight_uri="bioimage_zoo_model/bioimagezoo_weights_emdiffuse-n.pt",
        weight_type="torchscript",
        # the test input and output data as well as the description of the tensors
        # these are passed as list because we support multiple inputs / outputs per model
        test_inputs=["bioimage_zoo_model/test-input.npy"],
        test_outputs=["bioimage_zoo_model/test-output.npy"],
        input_axes=["bcyx"],
        output_axes=["bcyx"],
        # where to save the model zip, how to call the model and a short description of it
        output_path="bioimage_zoo_model/emdiffuse-n.zip",
        name="EMDiffuse-n",
        description="EMDiffuse-n denoising EM images",
        # additional metadata about authors, licenses, citation etc.
        authors=[{"name": "Chixiang Lu"}, {"name": "Xiaojuan Qi"}, {"name": "Haibo Jiang"}],
        license="CC-BY-4.0",
        maintainers = [{"github_user": "luchixiang", "name": "Chixiang Lu", "email": "luchixiang@gmail.com"}],
        documentation="bioimage_zoo_model/doc.md",
        tags=["EMDiffuse", "Electron microscope", "Denoising", "Diffusion Model", "Image restoration", ],  # the tags are used to make models more findable on the website
        cite=[{"text": "Lu et al.", "doi": "https://doi.org/10.1101/2023.07.12.548636"}],
    )
    from bioimageio.core.resource_tests import test_model
    import bioimageio.core
    my_model = bioimageio.core.load_resource_description("bioimage_zoo_model/emdiffuse-n.zip")
    test_model(my_model)
    phase_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/EMDiffuse-n.json',
                        help='JSON file for configuration')
    parser.add_argument('--path', type=str, default=None, help='patch of cropped patches')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-z', '--z_times', default=None, type=int)
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-mean', type=int, default=2)
    # parser.add_argument('-')

    ''' parser configs '''
    args = parser.parse_args()

    opt = Praser.parse(args)

    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids'])  # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:' + args.port
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1
        main_worker(0, 1, opt)
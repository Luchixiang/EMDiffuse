import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
import os
import shutil
import torch.nn as nn
import core.util as Util
from warmup_scheduler import GradualWarmupScheduler


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DiReP(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(DiReP, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None

        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()
        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.schedulers.append(GradualWarmupScheduler(self.optG, multiplier=1, total_epoch=20))
        self.resume_training()

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)
            self.netG_EMA.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.img_min = data.get('img_min', None)
        self.img_max = data.get('img_max', None)
        self.gt_min = data.get('gt_min', None)
        self.gt_max = data.get('gt_max', None)
        self.path = data['path']
        self.batch_size = len(data['path'])

    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu() + 1) / 2,
            'cond_image': (self.cond_image.detach()[:].float().cpu() + 1) / 2,
        }
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu() + 1) / 2
            })

        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.output[idx].detach().float().cpu())
            ret_path.append('Input_{}'.format(self.path[idx]))
            ret_result.append(self.cond_image[idx].detach().float().cpu())
            if self.mean > 1:
                for output_index in range(len(self.outputs)):
                    ret_path.append('Out_round{}_{}'.format(output_index, self.path[idx]))
                    ret_result.append(self.outputs[output_index][idx].detach().float().cpu())
        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in self.phase_loader:
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    print('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    self.output, self.visuals = self.netG.module.restoration(self.cond_image,
                                                                             sample_num=self.sample_num,
                                                                             y_0=self.gt_image)
                else:
                    self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num,
                                                                      y_0=self.gt_image)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results(), norm=self.opt['norm'])

        return self.val_metrics.result()

    def model_test(self, sample_num):
        if self.opt['distributed']:
            output, self.visuals = self.netG.module.restoration(self.cond_image,
                                                                sample_num=sample_num)
        else:

            output, self.visuals = self.netG.restoration(self.cond_image,
                                                         sample_num=sample_num)
        return output

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in self.phase_loader:
                self.set_input(phase_data)
                self.outputs = []
                for i in range(self.mean):
                    output = self.model_test(self.sample_num)
                    self.outputs.append(output)
                if self.mean > 1:
                    self.output = torch.stack(self.outputs, dim=0).mean(dim=0)
                    self.model_uncertainty = torch.stack(self.outputs, dim=0).std(dim=0)
                else:
                    self.output = self.outputs[0]
                    self.model_uncertainty = torch.zeros_like(self.output)
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results(), norm=self.opt['norm'])

        test_log = self.test_metrics.result()
        ''' save logged informations into log dict '''
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard '''
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label + '_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label + '_ema')
        self.save_training_state()

    def load_network(self, network, network_label, strict=True):
        if self.opt['path']['resume_state'] is None:
            return
        self.logger.info('Beign loading pretrained model [{:s}] ...'.format(network_label))

        model_path = "{}_{}.pth".format(self.opt['path']['resume_state'], network_label)

        if not os.path.exists(model_path):
            self.logger.warning('Pretrained model in [{:s}] is not existed, Skip it'.format(model_path))
            return

        self.logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: Util.set_device(storage)),
                                strict=strict)

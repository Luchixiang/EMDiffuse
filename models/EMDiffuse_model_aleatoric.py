import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
import os
import torch.nn as nn
import core.util as Util


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


class UncertaintyModel(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(UncertaintyModel, self).__init__(**kwargs)

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
        self.netG_copy = copy.deepcopy(self.netG)
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG_copy.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
            self.netG_copy.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG_copy.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)
            self.netG_copy.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        # self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.val_metrics = LogTracker(*['var_mean', 'var_min', 'var_max'], phase='val')
        self.test_metrics = LogTracker(*['var_mean', 'var_min', 'var_max'], phase='test')
        # self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')
        self.sample_num = sample_num
        self.task = task

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        # print(self.cond_image.min(), self.cond_image.max())
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
            ret_path.append('Input_{}'.format(self.path[idx]))
            ret_result.append(self.cond_image[idx].detach().float().cpu())
            ret_path.append('Variance_{}'.format(self.path[idx]))
            ret_result.append(torch.sqrt(torch.exp(self.variance[idx].detach().float().cpu())))
            # print('saving', torch.sqrt(torch.exp(self.predict_variance_out[idx].detach().float().cpu())).max(), torch.sqrt(torch.exp(self.predict_variance_out[idx].detach().float().cpu())).min())
        if self.task in ['inpainting', 'uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in self.phase_loader:
            self.set_input(train_data)
            self.optG.zero_grad()
            self.netG_copy.eval()
            with torch.no_grad():
                noise_hat, noise, sample_gammas = self.netG_copy(self.gt_image, self.cond_image, mask=self.mask)
            loss, _ = self.netG(self.gt_image, y_cond=self.cond_image, mask=self.mask, noise_hat=noise_hat, noise=noise, variance=True, sample_gammas=sample_gammas)
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
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    def val_step(self):
        self.netG.eval()
        self.netG_copy.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for phase_data in self.phase_loader:
                self.set_input(phase_data)
                variance_final = torch.zeros_like(self.gt_image)
                if self.opt['distributed']:
                    for times in [2, 4, 8, 16, 32]:
                        noise_hat, noise, sample_gammas = self.netG_copy.module(self.gt_image, self.cond_image, times=times)
                        loss, variance = self.netG.module(self.gt_image, y_cond=self.cond_image, noise_hat=noise_hat,
                                                          noise=noise, variance=True, sample_gammas=sample_gammas)
                        variance_final += variance
                else:
                    for times in [2, 4, 8, 16, 32]:
                        noise_hat, noise, sample_gammas = self.netG_copy(self.gt_image, self.cond_image, times=times)
                        loss, variance = self.netG(self.gt_image, y_cond=self.cond_image, noise_hat=noise_hat,
                                                          noise=noise, variance=True, sample_gammas=sample_gammas)
                        variance_final += variance
                variance = variance_final / len([2, 4, 8, 16, 32])
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                self.val_metrics.update('var_max', variance.max())
                self.val_metrics.update('var_min', variance.min())
                self.val_metrics.update('var_mean', variance.mean())
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')
        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in self.phase_loader:
                self.set_input(phase_data)
                variance_final = torch.zeros_like(self.gt_image)
                if self.opt['distributed']:
                    for times in [2, 4, 8, 16, 32]:
                        noise_hat, noise, sample_gammas = self.netG_copy.module(self.gt_image, self.cond_image,
                                                                                times=times)
                        loss, variance = self.netG.module(self.gt_image, y_cond=self.cond_image, noise_hat=noise_hat,
                                                          noise=noise, variance=True, sample_gammas=sample_gammas)
                        variance_final += variance
                else:
                    for times in [2, 4, 8, 16, 32]:
                        noise_hat, noise, sample_gammas = self.netG_copy(self.gt_image, self.cond_image, times=times)
                        loss, variance = self.netG(self.gt_image, y_cond=self.cond_image, noise_hat=noise_hat,
                                                   noise=noise, variance=True, sample_gammas=sample_gammas)
                        variance_final += variance
                self.variance = variance_final / len([2, 4, 8, 16, 32])
                self.iter += self.batch_size
                print(self.variance.mean())
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                self.test_metrics.update('var_max', self.variance.max())
                self.test_metrics.update('var_min', self.variance.min())
                self.test_metrics.update('var_mean', self.variance.mean())
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
        self.load_network(network=self.netG_copy, network_label=netG_label, strict=False)
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
        state_dict = torch.load(model_path, map_location=lambda storage, loc: Util.set_device(storage))
        model_dict = network.state_dict()
        model_dict.update(state_dict)
        network.load_state_dict(model_dict, strict=strict)

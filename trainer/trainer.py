import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.loss import MNSSLoss

class MNSSTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        criterion = MNSSLoss(config['scale_factor'], config['k'], config['w'])
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, [low_res_list, depth_list, motion_list, truth_list] in enumerate(self.data_loader):
            n = len(low_res_list)
            
            self.optimizer.zero_grad()
            
            prev_high_res = None
            average_loss = 0
            average_metrics = np.zeros(len(self.metric_ftns))
            for i in range(1, n):
                low_res = low_res_list[i].to(self.device)
                depth = depth_list[i].to(self.device)
                prev_depth = depth_list[i-1].to(self.device)
                motion = motion_list[i].to(self.device)
                truth = truth_list[i].to(self.device)
                if i > 1 or self.config['use_prev_high_res']:
                    prev_high_res = truth_list[i-1].to(self.device)
                jitter = (0, 0) # not-implemented yet

                img_ss, img_aa = self.model(low_res, depth, prev_high_res, prev_depth, motion, jitter)
                loss = self.criterion(img_aa, img_ss,  truth, jitter) / (n - 1) # average loss over all frames
                    
                loss.backward()

                average_loss += loss.item()
                for j, met in enumerate(self.metric_ftns):
                    average_metrics[j] += met(img_ss, truth).item() / (n - 1)
            
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for i, met in enumerate(self.metric_ftns):
                self.train_metrics.update(met.__name__, average_metrics[i])

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

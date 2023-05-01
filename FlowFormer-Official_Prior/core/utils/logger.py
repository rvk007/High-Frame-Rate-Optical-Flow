import wandb
from loguru import logger as loguru_logger
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, model, scheduler, cfg):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.cfg = cfg

    def _print_training_status(self):
        wandb.log({
            f'train_{k}': self.running_loss[k]/self.cfg.sum_freq for k in self.running_loss.keys()
        })
        wandb.log({
            'train_lr': self.scheduler.get_last_lr()[0],
            'train_steps': self.total_steps+1,
        })
        metrics_data = [self.running_loss[k]/self.cfg.sum_freq for k in (self.running_loss.keys())]
        training_str = "[{:6d}, {}] ".format(self.total_steps+1, self.scheduler.get_last_lr())
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        print(self.running_loss)

        for k in self.running_loss:
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.cfg.sum_freq == self.cfg.sum_freq-1:
            self._print_training_status()
            self.running_loss = {}

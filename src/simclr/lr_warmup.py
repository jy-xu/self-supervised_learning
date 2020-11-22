from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epoch, after_scheduler=None):
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(LinearWarmupScheduler, self).__init__(optimizer)
    
    def step(self):
        if self.finished and self.after_scheduler:
            self.after_scheduler.step()
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(LinearWarmupScheduler, self).step()
    
    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return self.base_lrs
        else:
            return [base_lr * (float(self.last_epoch) / self.warmup_epoch) for base_lr in self.base_lrs]
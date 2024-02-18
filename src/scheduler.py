"""
Implements WarmupDecayedCosineAnnealingWarmRestarts,
Run the file to see scheduler.png to see the lr graph. 
Basically linear warmup then cosine annealing with warm restarts.
"""

import torch
import math
import functools

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR


class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    """
    Subclass of regular cosine annealing with warm restarts to add decaying peak learning rates.
    Ref: https://stackoverflow.com/a/73747249
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, decay=1):
        super().__init__(optimizer, T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
        self.decay = decay
        self.initial_lrs = self.base_lrs

    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0

            self.base_lrs = [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

        super().step(epoch)


def _constant_warmup(iteration, warmup_iterations):
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier


def LinearWarmup(optimizer, T_warmup):
    _decay_func = functools.partial(_constant_warmup, warmup_iterations=T_warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def WarmupDecayedCosineAnnealingWarmRestarts(optimizer, warmup_iters, T_0=500, decay=0.8):
    """
    Adds a linear warmup in front of the decayed cosine annealing with warm restarts

    optimizer: self-explanatory
    warmup_iters: number of iterations for the linear warmup
    T_0: Number of iterations for the first restart
    decay: decay factor for each warm restart, restarted_peak_lr = prev_peak_lr * decay
    """
    warmup_scheduler = LinearWarmup(optimizer, T_warmup=warmup_iters)
    cos_scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=T_0, decay=decay)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cos_scheduler], milestones=[warmup_iters])
    return scheduler


if __name__ == "__main__":

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Dummy parameters
    parameters = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    total_iters = 2000
    warmup_iters = 100

    # Test
    optimizer = torch.optim.Adam([parameters], lr=0.2)
    scheduler = WarmupDecayedCosineAnnealingWarmRestarts(optimizer, warmup_iters, T_0=300, decay=0.8)

    actual_lr = []
    for _iter in range(total_iters):
        optimizer.step()
        scheduler.step()
        actual_lr.append(optimizer.param_groups[0]["lr"])
    plt.plot(list(range(total_iters)), actual_lr, "-", label="CosineAnnealingWarmRestartsDecay with Warmup")

    plt.xlabel("iterations")
    plt.ylabel("learning rate")
    plt.legend()
    plt.savefig("scheduler.png")

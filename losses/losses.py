import math
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim as MS_SSIM

class lambda_weighted_MSELoss(torch.nn.Module):
    def __init__(self):
        super(lambda_weighted_MSELoss, self).__init__()

    def forward(self, im0, im1, lambda_rd):
        square_diff = (im0 - im1) ** 2.0
        mse_batch = torch.mean(square_diff, (1, 2, 3))
        weighted_mse = mse_batch * lambda_rd[:, 0]
        train_mse = torch.mean(weighted_mse)
        return train_mse

class lambda_weighted_RateDistortionLoss(nn.Module):
    def __init__(self, metric='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.metric = metric

    def forward(self, output, target, lambda_rd):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out['bpp_loss'] = sum(
            (-torch.log2(likelihoods).sum() / num_pixels)
            for likelihoods in output['likelihoods'].values()
        )

        if self.metric == 'mse':
            out["mse_loss"] = self.mse(output['x_hat'], target)
            weighted_mse = out["mse_loss"]  * lambda_rd[:, 0]
            mseloss = torch.mean(weighted_mse)
            out["ms_ssim_loss"] = None
            out["loss"] = 255 ** 2 * mseloss + out["bpp_loss"]

        return out


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, metric='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metric = metric

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out['bpp_loss'] = sum(
            (-torch.log2(likelihoods).sum() / num_pixels)
            for likelihoods in output['likelihoods'].values()
        )

        if self.metric == 'mse':
            out['mse_loss'] = self.mse(output['x_hat'], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        elif self.metric == 'ms-ssim':
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - MS_SSIM(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]

        return out


class Metrics(nn.Module):
    def __init__(self):
        super().__init__()

    def MSE(self, x, y):
        # x, y: 4D [0, 1]
        return torch.mean((x - y) ** 2, dim=[1, 2, 3])

    def PSNR(self, x, y):
        # x, y: 4D [0, 1]
        mse = self.MSE(x, y)
        psnr = 10 * torch.log10(1. / mse)  # (B,)
        return torch.mean(psnr)

    def MS_SSIM(self, x, y):
        # x, y: 4D [0, 1]
        ms_ssim = MS_SSIM(x, y, data_range=1., size_average=True)  # (B,)
        return torch.mean(ms_ssim)

    def accuracy(self, output, target, topk=(1, 5)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W

        bpp = sum(
            (-torch.log2(likelihoods).sum() / num_pixels)
            for likelihoods in output['likelihoods'].values()
        )
        psnr = self.PSNR(output['x_hat'], target)
        ms_ssim = self.MS_SSIM(output['x_hat'], target)

        return bpp, psnr, ms_ssim

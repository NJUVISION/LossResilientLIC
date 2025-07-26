import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
from losses.losses import Metrics, RateDistortionLoss
from utils import *
from dataset import get_dataloader
from models.mbt_mean import MeanScaleHyperprior
from models.ProgDTD import ProgDTD
from models.LRLIC_wo_Prog import LRLICWoProg
from models.LRLIC_ConvNext_wo_Prog import LRLICConvNextWoProg
from models.LRLIC_w_Prog import LRLICWProg
from models.LRLIC_ConvNext_w_Prog import LRLICConvNextWProg


import logging
logging.getLogger('PIL').setLevel(logging.WARNING)


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, config):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": config['learning_rate']},
        "aux": {"type": "Adam", "lr": config['aux_learning_rate']},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model_name, model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )



def test_epoch(epoch, test_dataloader, model, criterion):
    metric = Metrics()
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            bpp, psnr, ms_ssim = metric(out_net, d)
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    print(f"bpp: {bpp:.3f} | PSNR: {psnr:.3f} | MS-SSIM: {ms_ssim:.3f}")

    return loss.avg



def main(argv):
    config = get_config(argv[0])
    print(config['model'])
    print(config['best_save_path'])
    # args = parse_args(argv)

    if config['seed'] != 'None':
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])


    device = "cuda" if config['cuda'] and torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader = get_dataloader(config)

    if config['model'] == "MeanScaleHyperprior":
        net = MeanScaleHyperprior(config)
        criterion = RateDistortionLoss(lmbda=config['lmbda'])
    elif config['model'] == "ProgDTD":
        net = ProgDTD(config)
        criterion = RateDistortionLoss(lmbda=config['lmbda'])
    elif config['model'] == "LRLICWoProg":
        net = LRLICWoProg(config)
        criterion = RateDistortionLoss(lmbda=config['lmbda'])
    elif config['model'] == "LRLICConvNextWoProg":
        net = LRLICConvNextWoProg(config)
        criterion = RateDistortionLoss(lmbda=config['lmbda'])
    elif config['model'] == "LRLICWProg":
        net = LRLICWProg(config)
        criterion = RateDistortionLoss(lmbda=config['lmbda'])
    elif config['model'] == "LRLICNextWProg":
        net = LRLICConvNextWProg(config)
        criterion = RateDistortionLoss(lmbda=config['lmbda'])
    else:
        raise ValueError(f"Model {config['model']} not defined.")

    net = net.to(device)

    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, config)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    last_epoch = 0
    if config['checkpoint']:  # load from previous checkpoint
        print("Loading", config['checkpoint'])
        checkpoint = torch.load(config['checkpoint'], map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # net.load_state_dict(checkpoint)

    best_loss = float("inf")
    for epoch in range(last_epoch, config['epochs']):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            config['model'],
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            config['clip_max_norm'],
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if config['save']:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                filename=os.path.join(config['save_path']),
                bestname=os.path.join(config['best_save_path']),
            )




if __name__ == "__main__":
    from models.utils import setup_seed
    setup_seed(1)
    main(sys.argv[1:])
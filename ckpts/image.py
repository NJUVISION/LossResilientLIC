import sys
import torch

sys.path.append("..")
from models.mbt_mean import MeanScaleHyperprior
from models.ProgDTD import ProgDTD
from models.LRLIC_ConvNext_w_Prog import LRLICConvNextWProg
from models.LRLIC_w_Prog import LRLICWProg
from models.LRLIC_wo_Prog import LRLICWoProg
from models.LRLIC_ConvNext_wo_Prog import LRLICConvNextWoProg
from utils import get_config
from torch.hub import load_state_dict_from_url
from .pretrained import load_pretrained


__all__ = [
    "MeanScaleHyperprior",
    "ProgDTD",
    "LRLICWoProg",
    "LRLICConvNextWoProg",
    "LRLICWProg",
    "LRLICConvNextWProg",
]
model_architectures = {
    "MeanScaleHyperprior": MeanScaleHyperprior,
    "ProgDTD": ProgDTD,
    "LRLICWoProg": LRLICWoProg,
    "LRLICConvNextWoProg": LRLICConvNextWoProg,
    "LRLICWProg": LRLICWProg,
    "LRLICConvNextWProg": LRLICConvNextWProg,
}

models = {
    "MeanScaleHyperprior": MeanScaleHyperprior,
    "ProgDTD": ProgDTD,
    "LRLICWoProg": LRLICWoProg,
    "LRLICConvNextWoProg": LRLICConvNextWoProg,
    "LRLICWProg": LRLICWProg,
    "LRLICConvNextWProg": LRLICConvNextWProg,
}

root_url = "./ckpts"
model_urls = {
    "MeanScaleHyperprior": {
        "mse": {
            1: f"/workspace/shw/PLIC/raw_model/raw_q1_best.pth.tar",
            2: f"/workspace/shw/PLIC/raw_model/raw_q2_best.pth.tar",
            3: f"/workspace/shw/PLIC/raw_model/raw_q3_best.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/",
        },
    },

    "ProgDTD": {
        "mse": {
            1: f"/workspace/shw/PLIC/onlyprog_model/q1_best.pth.tar",
            2: f"/workspace/shw/PLIC/onlyprog_model/q2_best.pth.tar",
            3: f"/workspace/shw/PLIC/onlyprog_model/q3_best.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/",
        },
    },

    "LRLICWoProg": {
        "mse": {
            1: f"/workspace/shw/PLIC/allnop_model/mean_q1_best.pth.tar",
            2: f"/workspace/shw/PLIC/allnop_model/mean_q2_best.pth.tar",
            3: f"/workspace/shw/PLIC/allnop_model/mean_q3_best.pth.tar",
            4: f"/workspace/shw/PLIC/allnop_model/gil_q1_best.pth.tar",
            5: f"/workspace/shw/PLIC/allnop_model/gil_q2_best.pth.tar",
            6: f"/workspace/shw/PLIC/allnop_model/gil_q3_best.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/",
        },
    },


    "LRLICConvNextWoProg": {
        "mse": {
            1: f"/workspace/shw/PLIC/nopnext_model/rand_q1_best.pth.tar",
            2: f"/workspace/shw/PLIC/nopnext_model/rand_q2_best.pth.tar",
            3: f"/workspace/shw/PLIC/nopnext_model/rand_q3_best.pth.tar",
            4: f"/workspace/shw/PLIC/nopnext_model/gil_q1_best.pth.tar",
            5: f"/workspace/shw/PLIC/nopnext_model/gil_q2_best.pth.tar",
            6: f"/workspace/shw/PLIC/nopnext_model/gil_q3_best.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/",
        },
    },

    "LRLICWProg": {
        "mse": {
            1: f"/workspace/shw/PLIC/mask_model/cat4channel_q1_best.pth.tar",
            2: f"/workspace/shw/PLIC/mask_model/cat4channel_q2_best.pth.tar",
            3: f"/workspace/shw/PLIC/mask_model/cat4channel_q3_best.pth.tar",
            4: f"/workspace/shw/PLIC/gilbert_model/q1_best.pth.tar",
            5: f"/workspace/shw/PLIC/gilbert_model/q2_best.pth.tar",
            6: f"/workspace/shw/PLIC/gilbert_model/q3_best.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/",
        },
    },


    "LRLICConvNextWProg": {
        "mse": {
            1: f"/workspace/shw/PLIC/next_model/rand_next_q1_best.pth.tar",
            2: f"/workspace/shw/PLIC/next_model/rand_next_q2_best.pth.tar",
            3: f"/workspace/shw/PLIC/next_model/rand_next_q3_best.pth.tar",
            4: f"/workspace/shw/PLIC/next_model/gil_next_q1_best.pth.tar",
            5: f"/workspace/shw/PLIC/next_model/gil_next_q2_best.pth.tar",
            6: f"/workspace/shw/PLIC/next_model/gil_next_q3_best.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/",
        },
    },

}


cfgs = {
    "MeanScaleHyperprior": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
    },

    "ProgDTD": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192), },

    "LRLICWoProg": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (128, 192), },

    "LRLICConvNextWoProg": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (128, 192), },

    "LRLICWProg": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (128, 192), },

    "LRLICConvNextWProg": {
        1: (128, 192),
        2: (128, 192),
        3: (128, 192),
        4: (128, 192),
        5: (128, 192),
        6: (128, 192), },
}


def _load_model(
        architecture, config, metric, quality, pretrained=False, **kwargs
):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')
    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        if (
                architecture not in model_urls
                or metric not in model_urls[architecture]
                or quality not in model_urls[architecture][metric]
        ):
            raise RuntimeError("Pre-trained model not yet available")

        url = model_urls[architecture][metric][quality]
        # print("Loading Ckpts From:", url)
        # state_dict = load_state_dict_from_url(url, progress=progress)
        state_dict = torch.load(url)
        # state_dict = load_pretrained(state_dict)

        # config = get_config("codec_config.yaml")
        config['N'] = cfgs[architecture][quality][0]
        config['M'] = cfgs[architecture][quality][1]
        model = model_architectures[architecture](config)
        # print(state_dict.keys())
        # model.load_state_dict(state_dict['model'])
        model.load_state_dict(state_dict['state_dict'])

        # TODO: should be put in traning loop
        model.update()

        # model = model_architectures[architecture].from_state_dict(state_dict)

    # model = model_architectures[architecture](*cfgs[architecture][quality], **kwargs)
    return model


def MeanScaleHyperprior(quality, config, metric="mse", pretrained=False, **kwargs):
    r"""
    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
    """
    if metric not in ("mse",):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("MeanScaleHyperprior", config, metric, quality, pretrained, **kwargs)

def  ProgDTD(quality, config, metric="mse", pretrained=False, **kwargs):
    r"""
    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
    """
    if metric not in ("mse",):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("ProgDTD", config, metric, quality, pretrained, **kwargs)


def  LRLICWoProg(quality, config, metric="mse", pretrained=False, **kwargs):
    r"""
    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
    """
    if metric not in ("mse",):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("LRLICWoProg", config, metric, quality, pretrained, **kwargs)


def  LRLICConvNextWoProg(quality, config, metric="mse", pretrained=False, **kwargs):
    r"""
    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
    """
    if metric not in ("mse",):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("LRLICConvNextWoProg", config, metric, quality, pretrained, **kwargs)

def  LRLICWProg(quality, config, metric="mse", pretrained=False, **kwargs):
    r"""
    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
    """
    if metric not in ("mse",):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("LRLICWProg", config, metric, quality, pretrained, **kwargs)


def LRLICConvNextWProg(quality, config, metric="mse", pretrained=False, **kwargs):
    r"""
    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
    """
    if metric not in ("mse",):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("LRLICConvNextWProg", config, metric, quality, pretrained, **kwargs)






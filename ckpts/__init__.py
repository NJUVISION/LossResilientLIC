import sys
from .image import *
from .pretrained import load_pretrained as load_state_dict

models = {
    "MeanScaleHyperprior": MeanScaleHyperprior,
    "ProgDTD": ProgDTD,
    "LRLICWoProg": LRLICWoProg,
    "LRLICConvNextWoProg": LRLICConvNextWoProg,
    "LRLICWProg": LRLICWProg,
    "LRLICConvNextWProg": LRLICConvNextWProg,
}
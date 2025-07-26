# LossResilientLIC

An implementation of "Towards Loss-Resilient Image Coding for Unstable Satellite Networks" (**AAAI 2025 Oral**).

------

## Abstract

Our method builds on the channel-wise progressive coding framework, incorporating Spatial-Channel Rearrangement (SCR) on the encoder side and Mask Conditional Aggregation (MCA) on the decoder side to improve reconstruction quality with unpredictable errors. By integrating the Gilbert-Elliot model into the training process, we enhance the model’s ability to generalize in real-world network conditions.

------

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{sha2025towards,
  title={Towards Loss-Resilient Image Coding for Unstable Satellite Networks},
  author={Sha, Hongwei and Dong, Muchen and Luo, Quanyou and Lu, Ming and Chen, Hao and Ma, Zhan},
  journal={arXiv preprint arXiv:2501.11263},
  year={2025}
}
```

------

## Installation

1. Clone the repo

   ```bash
   git clone https://github.com/NJUVISION/LossResilientLIC.git
   cd LossResilientLIC
   ```

2. Install dependencies

   ```bash
   pip install compressai
   pip install timm
   ```

### Training

```bash
python train.py train_config.yaml
```

### Evaluation

```bash
python eval_channel_packet.py
```

### Progressive Inference

```bash
python progressive_test.py 
```

------

## Directory Structure

```text
├── bin/                # Compressed bit stream
├── exp/                # Organized network data
├── losses/                # RD loss
├── models/                 # Model definitions
├── rec_img/                # Decompressed images
├── sim2net/                 # Simple Network Simulator
├── train.py                # Training entrypoint
├── eval_channel_packet.py             # Evaluation entrypoint with packet
├── codec.py            # Real inference Codec
├── train_config.yaml            # Training config
├── codec_config.yaml            # Real inference Codec config
├── progressive_test.py        # Progressive inference entrypoint
└── README.md
```

------

## License

BSD 3-Clause Clear License

------

## Acknowledgements

This work builds upon the [CompressAI](https://github.com/InterDigitalInc/CompressAI?tab=readme-ov-file), [sim2net](https://github.com/mkalewski/sim2net) and [ProgDTD](https://github.com/ds-kiel/ProgDTD).

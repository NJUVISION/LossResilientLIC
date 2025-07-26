import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import struct
import time
import numpy as np
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import compressai
from ckpts import models
from utils import get_config



def BoolConvert(a):
    b = [False, True]
    return b[int(a)]


def Average(lst):
    return sum(lst) / len(lst)


def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    metric = metric_ids[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)
    return model_ids[model_name], code


def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    model_id, code = header
    quality = (code & 0x0F) + 1
    metric = code >> 4
    return (
        inverse_dict(model_ids)[model_id],
        inverse_dict(metric_ids)[metric],
        quality,
    )


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt

def get_path_list(path):
    if os.path.isdir(path):
        file_list = os.listdir(path)
        file_list.sort()
    else:
        file_list = [path]
    return file_list

def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def _encode(image, model, metric, quality, coder, output, config):
    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    img = load_image(image)
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models[model](quality=quality, config=config, metric=metric, pretrained=True).to(device).eval()
    # if config['model'] != "MeanScaleHyperprior":
    net.set_progressive_level(progressiveness=config['progressiveness'], p_hyper=config['p_hyper'], p = config['p_latent'])
    load_time = time.time() - start

    x = img2torch(img)
    h, w = x.size(2), x.size(3)
    p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
    x = pad(x, p)

    x = x.to(device)
    # net = net.cuda()
    with torch.no_grad():
        out = net.compress_packets(x, config['max_packet_length'], config["quality"])

    shape = out["shape"]
    header = get_header(model, metric, quality)

    with Path(output).open("wb") as f:
        write_uchars(f, header)
        # write original image size
        write_uints(f, (h, w))
        # write shape and number of encoded latents
        write_body(f, shape, out["packets"])

    enc_time = time.time() - enc_start
    size = filesize(output)
    bpp = float(size) * 8 / (img.size[0] * img.size[1])
    # print(
    #     f"{bpp:.3f} bpp |"
    #     f" Encoded in {enc_time:.2f}s (model loading: {load_time:.2f}s)"
    # )


def _decode(inputpath, config, coder, show, output=None):
    compressai.set_entropy_coder(coder)

    with Path(inputpath).open("rb") as f:
        model, metric, quality = parse_header(read_uchars(f, 2))
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models[model](quality=quality, config = config, metric=metric, pretrained=True).to(device).eval()
    # if config['model'] != "MeanScaleHyperprior":
    # net.set_loss_packet(config['loss_packet'], config['packet_method'], config['packet_loss_ratio'])
    net.set_progressive_level(progressiveness=config['progressiveness'], p_hyper=config['p_hyper'],
                              p=config['p_latent'])
    torch.cuda.synchronize()
    start = time.time()
    # for decode_times in range(100):
    with torch.no_grad():
        out = net.decompress_packets(strings, shape)

        x_hat = crop(out["x_hat"], original_size)
        img = torch2img(x_hat)
    torch.cuda.synchronize()
    end = time.time()
    dec_time = end - start
    # print(f"Decoded in {dec_time:.2f}s")

    if show:
        show_image(img)
    if output is not None:
        img.save(output)


def show_image(img: Image.Image):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.title.set_text("Decoded image")
    ax.imshow(img)
    fig.tight_layout()
    plt.show()


def encode(filename, config, output):
    _encode(filename, config["model"], "mse", int(config['quality']), compressai.available_entropy_coders()[0], output, config)


def decode(binname, recname, config):
    _decode(binname, config, compressai.available_entropy_coders()[0], False, recname)

def compute_psnr(Imagepath1, Imagepath2):
    img1 = np.array(Image.open(Imagepath1))
    img2 = np.array(Image.open(Imagepath2))
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compute_bpp(binfile, pixelnum):
    return os.path.getsize((binfile)) * 8/ pixelnum




if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    model_ids = {k: i for i, k in enumerate(models.keys())}
    metric_ids = {"mse": 0, "ms-ssim": 1}
    # config = get_config(sys.argv[1:][0])
    config = get_config("codec_config.yaml")
    dir_path = r"/workspace/shared/Kodak/kodim21.png"
    rec_dir = "./rec_img"
    torch.set_num_threads(1)  # just to be sure

    file_list = get_path_list(dir_path)
    binfile = 'out.bin'
    recfile = 'out.png'

    logfile = 'log.txt'
    print(config['model'])
    mean_bpp = 0
    mean_psnr = 0
    with open(logfile, 'w') as f:
        for file_name in file_list:
            if os.path.isdir(dir_path):
                file_name = dir_path + file_name
            print("processing " + file_name)
            h, w = Image.open(file_name).size
            pixel_num = h * w
            encode(file_name, config, output=binfile)
            decode(binfile, recfile, config)

            bpp = compute_bpp(binfile, pixel_num)
            psnr = compute_psnr(file_name, recfile)
            mean_bpp += bpp
            mean_psnr += psnr
            f.write(file_name+'\t'+str(bpp)+'\t'+str(psnr)+'\n')
    mean_bpp = mean_bpp / len(file_list)
    mean_psnr = mean_psnr / len(file_list)
    print("mean", mean_bpp, "\t", mean_psnr)
    with open(logfile, 'a') as f:
        f.write("meanbppandpsnr" + '\t' + str(mean_bpp) + '\t' + str(mean_psnr) + '\n')

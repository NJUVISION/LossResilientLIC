import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import yaml


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def gaussian_kernel1d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """1D Gaussian kernel."""
    khalf = (kernel_size - 1) / 2.0
    x = torch.linspace(-khalf, khalf, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def gaussian_kernel2d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """2D Gaussian kernel."""
    kernel = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    return torch.mm(kernel[:, None], kernel[None, :])


def gaussian_blur(x, kernel=None, kernel_size=None, sigma=None):
    """Apply a 2D gaussian blur on a given image tensor."""
    if kernel is None:
        if kernel_size is None or sigma is None:
            raise RuntimeError("Missing kernel_size or sigma parameters")
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        device = x.device
        kernel = gaussian_kernel2d(kernel_size, sigma, device, dtype)

    padding = kernel.size(0) // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    x = torch.nn.functional.conv2d(
        x,
        kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1)),
        groups=x.size(1),
    )
    return x


def meshgrid2d(N: int, C: int, H: int, W: int, device: torch.device):
    """Create a 2D meshgrid for interpolation."""
    theta = torch.eye(2, 3, device=device).unsqueeze(0).expand(N, 2, 3)
    return F.affine_grid(theta, (N, C, H, W), align_corners=False)

def setup_seed(seed=1):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

def get_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        return config




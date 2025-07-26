import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.registry import register_model
import numpy as np
from .utils import setup_seed
from .layers import ConvNeXtLayer

from .base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from .utils import conv, deconv
import random
from sim2net.packet_loss.gilbert_elliott import GilbertElliott
from utils import get_config
from scipy.optimize import curve_fit

dataset_total_packet = 0
config = get_config("codec_config.yaml")
with open(config["channel"], 'r') as f:
    gilbert_channel = f.read()
    gilbert_channel = eval(gilbert_channel)
packet_offset = 0
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def Gilbert_Elliott_Channel_Zeroing(p, tensor):
    prhks = [
        [0.378563411896744, 0.883314627759071, 0.810000000000000, 0.938571428571429],
    ]  # for train
    prhk = random.choice(prhks)
    tensor_copy = tensor.clone()
    channel = tensor.shape[0]
    mask = torch.ones(channel, device=tensor.device)
    ge = GilbertElliott(prhk)

    if p != 1:
        keeping_channel = int(channel * p) + 1
        for i in range(keeping_channel):
            pl = int(ge.packet_loss())
            if pl == 1:
                mask[i] = 0
                tensor_copy[i, :, :] = 0
    else:
        for i in range(channel):
            pl = int(ge.packet_loss())
            if pl == 1:
                mask[i] = 0
                tensor_copy[i, :, :] = 0
    return tensor_copy, mask.float()



@register_model("bmshj2018-hyperprior")
class ScaleHyperprior(CompressionModel):
    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)
        depths = [2, 4, 6, 2, 2, 2]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.g_a = nn.Sequential(
            ConvNeXtLayer(dim_in=3,
                          dim_out=N,
                          depth=depths[0],
                          drop=dpr[sum(depths[:0]):sum(depths[:1])],
                          is_first=True,
                          encode=True),
            ConvNeXtLayer(dim_in=N,
                                      dim_out=N,
                                      depth=depths[1],
                                      drop=dpr[sum(depths[:1]):sum(depths[:2])],
                                      is_first=False,
                                      encode=True),
            ConvNeXtLayer(dim_in=N,
                          dim_out=N,
                          depth=depths[2],
                          drop=dpr[sum(depths[:2]):sum(depths[:3])],
                          is_first=False,
                          encode=True),
            ConvNeXtLayer(dim_in=N,
                          dim_out=M,
                          depth=depths[3],
                          drop=dpr[sum(depths[:3]):sum(depths[:4])],
                          is_first=False,
                          encode=True),
        )

        depths = depths[::-1]

        self.g_s = nn.Sequential(
            ConvNeXtLayer(dim_in=M,
                          dim_out=N,
                          depth=depths[2],
                          drop=dpr[sum(depths[:2]):sum(depths[:3])],
                          is_first=False,
                          encode=False),
            ConvNeXtLayer(dim_in=N,
                          dim_out=N,
                          depth=depths[3],
                          drop=dpr[sum(depths[:3]):sum(depths[:4])],
                          is_first=False,
                          encode=False),
            ConvNeXtLayer(dim_in=N,
                          dim_out=N,
                          depth=depths[4],
                          drop=dpr[sum(depths[:4]):sum(depths[:5])],
                          is_first=False,
                          encode=False),
            ConvNeXtLayer(dim_in=N,
                          dim_out=3,
                          depth=depths[5],
                          drop=dpr[sum(depths[:5]):sum(depths[:6])],
                          is_first=True,
                          encode=False),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


# @register_model("mbt2018-mean")
class LRLICConvNextWProg(ScaleHyperprior):
    def __init__(self, config, **kwargs):
        N = config['N']
        M = config['M']
        self.M = M
        super().__init__(N=N, M=M, **kwargs)

        self.progressiveness = config['progressiveness']
        self.progressiveness_range = config['progressiveness_range']
        self.p_hyper_latent = None
        self.p_latent = None

        self.loss_packet = config['loss_packet']


        self.g_conv_s = nn.Sequential(
            conv(M , 64, stride=1, kernel_size=3),
            nn.GELU(),
            conv(64, 4, stride=1, kernel_size=3),
            nn.GELU(),
        )

        self.g_mask_s = nn.Sequential(
            # deconv(2 * M, M),
            conv(M + 4, M, stride=1, kernel_size=3),
            nn.GELU(),
        )

        depths = [2, 4, 6, 2, 2, 2]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.h_a = nn.Sequential(
            ConvNeXtLayer(dim_in=M,
                          dim_out=N,
                          depth=depths[4],
                          drop=dpr[sum(depths[:4]):sum(depths[:5])],
                          is_first=False,
                          encode=True,
                          is_hyper=True),
            ConvNeXtLayer(dim_in=N,
                          dim_out=N,
                          depth=depths[5],
                          drop=dpr[sum(depths[:5]):sum(depths[:6])],
                          is_first=False,
                          encode=True,
                          is_hyper=True),
        )

        depths = depths[::-1]
        self.h_s = nn.Sequential(
            ConvNeXtLayer(dim_in=N,
                          dim_out=N,
                          depth=depths[0],
                          drop=dpr[sum(depths[:0]):sum(depths[:1])],
                          is_first=False,
                          encode=False,
                          is_hyper=True),
            ConvNeXtLayer(dim_in=N,
                          dim_out=M * 2,
                          depth=depths[1],
                          drop=dpr[sum(depths[:1]):sum(depths[:2])],
                          is_first=False,
                          encode=False,
                          is_hyper=True),
        )


    def set_progressive_level(self, progressiveness,  p_hyper, p):
        self.progressiveness = progressiveness
        self.p_hyper_latent = p_hyper
        self.p_latent = p
        print(f"progressiveness: {self.progressiveness}, p_hyper: {self.p_hyper_latent}, p_latent: {self.p_latent}")


    def set_loss_packet(self, loss_packet, packet_method, packet_loss_ratio):
        self.loss_packet = loss_packet
        self.packet_method = packet_method
        self.packet_loss_ratio = packet_loss_ratio
        print(f"loss_packet: {self.loss_packet}, packet_method: {self.packet_method}, packet_loss_rate: {self.packet_loss_ratio}")


    def rate_less_latent(self, data):
        self.save_p = []
        temp_data = data.clone()
        for i in range(data.shape[0]):
            if self.p_latent:
                # p shows the percentage of keeping
                p = self.p_latent
            else:
                p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1], 1)[0]
                self.save_p.append(p)

            if p == 1.0:
                pass
            else:
                p = int(p * data.shape[1])
                replace_tensor = torch.rand(data.shape[1] - p - 1, data.shape[2], data.shape[3]).fill_(0)

                if replace_tensor.shape[0] > 0:
                    temp_data[i, -replace_tensor.shape[0]:, :, :] = replace_tensor
        if not self.p_latent:
            return self.save_p, temp_data
        else:
            return self.p_latent, temp_data

    def rate_less_hyper_latent(self, data):
        temp_data = data.clone()
        for i in range(data.shape[0]):
            if self.p_hyper_latent:
                # p shows the percentage of keeping
                p = self.p_hyper_latent
            else:
                p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1], 1)[0]
                p = self.save_p[i]
            if p == 1.0:
                pass

            else:
                p = int(p * data.shape[1])
                replace_tensor = torch.rand(data.shape[1] - p - 1, data.shape[2], data.shape[3]).fill_(0)

                if replace_tensor.shape[0] > 0:
                    temp_data[i, -replace_tensor.shape[0]:, :, :] = replace_tensor

        return temp_data

    def drop_zeros_likelihood(self, likelihood, replace):
        temp_data = likelihood.clone()
        temp_data = torch.where(
            replace == 0.0,
            torch.cuda.FloatTensor([1.0])[0],
            likelihood,
        )
        return temp_data

    def random_channel_zeroing_prog(self, p, tensor, zero_prob_vector=[0.01, 0.03, 0.05, 0.07, 0.1]):
        rand_num = np.random.rand()
        if rand_num <= 0.5:
            mask = torch.ones(tensor.shape[0]).to(tensor.device)
            return tensor, mask
        else:
            if p != 1:
                p = int(tensor.shape[0] * p) + 1
                zero_prob = np.random.choice(zero_prob_vector)
                random_array = np.random.rand(p)
                channel_replace = tensor.shape[0] - p
                random_array_replace = np.ones(channel_replace)
                random_array = np.concatenate((random_array, random_array_replace), axis=0)
                tensor_copy = tensor.clone()
                tensor_copy[random_array < zero_prob, :, :] = 0
                mask = torch.from_numpy(random_array >= zero_prob).float()
                return tensor_copy, mask.to(tensor.device)
            else:
                zero_prob = np.random.choice(zero_prob_vector)
                random_array = np.random.rand(tensor.shape[0])
                tensor_copy = tensor.clone()
                tensor_copy[random_array < zero_prob, :, :] = 0
                mask = torch.from_numpy(random_array >= zero_prob).float()
                return tensor_copy, mask.to(tensor.device)



    def forward(self, x):
        y = self.g_a(x)
        # rearrange y
        y_copy = y.clone()
        y_chunk = y_copy.chunk(self.M // 4, dim=1)
        chanel_p = 0
        for t_y in y_chunk:
            t_swaped = swap_element(t_y)
            y[:, chanel_p:chanel_p + 4, :, :] = t_swaped
            chanel_p += 4

        z = self.h_a(y)

        if self.progressiveness:
            save_p, y = self.rate_less_latent(y)
            z = self.rate_less_hyper_latent(z)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        if self.progressiveness:
            y_likelihoods = self.drop_zeros_likelihood(y_likelihoods, y)
            z_likelihoods = self.drop_zeros_likelihood(z_likelihoods, z)

        if self.loss_packet:
            y_hat_random_zero = torch.zeros_like(y_hat)
            batch_mask = torch.ones(y_hat.shape[0], y_hat.shape[1]).to(y_hat.device)
            for i in range(y_hat.shape[0]):
                # y_hat_random_zero[i, :, :, :], mask = self.random_channel_zeroing_prog(save_p[i], y_hat[i, :, :, :])
                y_hat_random_zero[i, :, :, :], mask = Gilbert_Elliott_Channel_Zeroing(save_p[i], y_hat[i, :, :, :])
                batch_mask[i, :] = mask
            batch_mask = batch_mask.unsqueeze(2).unsqueeze(3).expand(y_hat.shape[0], y_hat.shape[1], y_hat.shape[2], y_hat.shape[3]).clone()
            batch_mask_copy = batch_mask.clone()
            batch_mask_chunk = batch_mask_copy.chunk(self.M // 4, dim=1)
            chanel_p = 0
            for t_mask in batch_mask_chunk:
                t_inv_mask = inv_swap_element(t_mask)
                batch_mask[:, chanel_p:chanel_p + 4, :, :] = t_inv_mask
                chanel_p += 4
            y_hat_random_zero_copy = y_hat_random_zero.clone()
            y_hat_zero_chunk = y_hat_random_zero_copy.chunk(self.M // 4, dim=1)
            chanel_p = 0
            for t_swaped in y_hat_zero_chunk:
                t_inv_swaped = inv_swap_element(t_swaped)
                y_hat_random_zero[:, chanel_p:chanel_p + 4, :, :] = t_inv_swaped
                chanel_p += 4
            conv_batch_mask = self.g_conv_s(batch_mask)
            mask_and_y_hat = torch.cat((y_hat_random_zero, conv_batch_mask), 1)
            mask_guidance_y = self.g_mask_s(mask_and_y_hat)
            x_hat = self.g_s(mask_guidance_y)
        else:
            x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y_copy = y.clone()
        y_chunk = y_copy.chunk(self.M // 4, dim=1)
        chanel_p = 0
        for t_y in y_chunk:
            t_swaped = swap_element(t_y)
            y[:, chanel_p:chanel_p + 4, :, :] = t_swaped
            chanel_p += 4
        z = self.h_a(y)
        print(f"keeping {int(self.p_latent * y.shape[1]) + 1} channel")
        if self.progressiveness:
            _, y = self.rate_less_latent(y)
            z = self.rate_less_hyper_latent(z)
            if self.p_latent != 1:
                p = int(self.p_latent * y.shape[1]) + 1
                y = y[:, :p, :, :]

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        if self.p_latent != 1:
            scales_hat = scales_hat[:, :p, :, :]
            means_hat = means_hat[:, :p, :, :]

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        # # length = 0
        for i in range(y.shape[1]):
            ith_channel_string = self.gaussian_conditional.compress(y[:, i, :, :], indexes[:, i, :, :], means=means_hat[:, i, :, :])
            print(len(ith_channel_string[0]))
            # length += len(ith_channel_string[0])
        # print(length)
        # print(len(y_strings[0]))

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}




    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = torch.zeros(1, self.M , shape[0] * 4, shape[1] * 4, device=z_hat.device)
        if self.p_latent != 1:
            p = int(self.p_latent * y_hat.shape[1]) + 1
            scales_hat = scales_hat[:, :p, :, :]
            means_hat = means_hat[:, :p, :, :]

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat_p = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        if self.loss_packet:
            # random drop channel
            # drop_channel_index = [5, 12, 22, 24, 33, 44, 64, 65, 75, 83, 88, 109, 121, 141, 149, 153, 155]
            drop_channel_index = []
            batch_mask = torch.ones(y_hat_p.shape[:2]).to(y_hat_p.device)
            batch_mask[:, drop_channel_index] = 0
            batch_mask = torch.cat((batch_mask, torch.ones(y_hat_p.shape[0], y_hat.shape[1] - y_hat_p.shape[1]).to(y_hat.device)), 1)
            y_hat_p[:, drop_channel_index, :, :] = 0
            batch_mask = batch_mask.unsqueeze(2).unsqueeze(3).expand(y_hat.shape).clone()
            batch_mask_copy = batch_mask.clone()
            batch_mask_chunk = batch_mask_copy.chunk(self.M // 4, dim=1)
            chanel_p = 0
            for t_mask in batch_mask_chunk:
                t_inv_mask = inv_swap_element(t_mask)
                batch_mask[:, chanel_p:chanel_p + 4, :, :] = t_inv_mask
                chanel_p += 4
            print(f"drop channel: {drop_channel_index}")
        if self.p_latent != 1:
            y_hat[:, :p, :, :] = y_hat_p
        else:
            y_hat = y_hat_p

        y_hat_random_zero_copy = y_hat.clone()
        y_hat_zero_chunk = y_hat_random_zero_copy.chunk(self.M // 4, dim=1)
        chanel_p = 0
        for t_swaped in y_hat_zero_chunk:
            t_inv_swaped = inv_swap_element(t_swaped)
            y_hat[:, chanel_p:chanel_p + 4, :, :] = t_inv_swaped
            chanel_p += 4
        conv_batch_mask = self.g_conv_s(batch_mask)
        mask_and_y_hat = torch.cat((y_hat, conv_batch_mask), 1)
        mask_guidance_y = self.g_mask_s(mask_and_y_hat)
        x_hat = self.g_s(mask_guidance_y).clamp_(0, 1)
        return {"x_hat": x_hat}


    def get_template_channel_bytes(self, y, index, means, q, max_packet_length):
        template_channel_bytes = []
        if q == 1:
            zero_info_channel = 48
            template_channel_index = [0, 4, 8, 16, 36, 44]
        elif q == 2:
            zero_info_channel = 60
            template_channel_index = [0, 4, 8,  36, 56]
        elif q == 3:
            zero_info_channel = 76
            template_channel_index = [0, 4, 8, 36, 72]
        elif q == 4:
            zero_info_channel = 48
            template_channel_index = [0, 4, 8, 16, 36, 44]
        elif q == 5:
            zero_info_channel = 68
            template_channel_index = [0, 4, 8, 36, 56, 60]
        elif q == 6:
            zero_info_channel = 76
            template_channel_index = [0, 4, 8, 36, 72]
        else:
            raise NotImplementedError
        for i in template_channel_index:
            template_channel_bytes.append(
                len(self.gaussian_conditional.compress(y[:, i, :, :], index[:, i, :, :], means=means[:, i, :, :])[0]))
        assert template_channel_bytes[0] <= max_packet_length - 4
        popt, pcov = curve_fit(exp_decay, template_channel_index, template_channel_bytes, p0=(450, 0.1, 0))
        x_fit = np.linspace(0, zero_info_channel - 1, zero_info_channel) # 只生成有信息量的channel
        y_fit = exp_decay(x_fit, *popt)
        y_fit = np.concatenate((y_fit, np.ones(self.M - zero_info_channel)))
        return np.array(y_fit, dtype=np.int32)

    def get_packet_channel(self, channel_lengths, max_packet_length):
        packet_channel_combination = []
        channel_combination = []
        current_packet_length = 0
        list_len = len(channel_lengths)
        for i in range(list_len):
            if current_packet_length + channel_lengths[i] <= max_packet_length:
                channel_combination.append(i)
                current_packet_length += channel_lengths[i]
            else:
                packet_channel_combination.append(channel_combination)
                channel_combination = [i]
                current_packet_length = channel_lengths[i]
            if i == list_len - 1:
                packet_channel_combination.append(channel_combination)
        print(len(packet_channel_combination), "estimate packet")
        global dataset_total_packet
        dataset_total_packet += len(packet_channel_combination)
        return packet_channel_combination

    # if the current packet length exceeds the maximum packet length,
    # then split the current packet into two packets
    def final_packets(self, packet_channel_combination, y, indexes, means_hat, max_packet_length):
        packets = []
        the_last_channel = packet_channel_combination[-1][-1] + 1
        for idx, one_packet_channel in enumerate(packet_channel_combination):
            start_channel = one_packet_channel[0]
            count_channel = len(one_packet_channel)
            if (start_channel + count_channel) % 4 == 0 and (
                    start_channel + count_channel) != the_last_channel and count_channel != 1:
                packet_channel_combination[idx + 1].insert(0, one_packet_channel[-1])
                one_packet_channel = one_packet_channel[:-1]
                packet_channel_combination[idx] = one_packet_channel
                count_channel = len(one_packet_channel)

            temp_packet_string = \
            self.gaussian_conditional.compress(y[:, start_channel:start_channel + count_channel, :, :],
                                               indexes[:, start_channel:start_channel + count_channel, :, :],
                                               means=means_hat[:, start_channel:start_channel + count_channel, :, :])[0]
            while (len(temp_packet_string) > max_packet_length - 4):
                count_channel = math.ceil(count_channel / 2)
                packet_channel_combination.insert(idx + 1, one_packet_channel[count_channel:])
                one_packet_channel = one_packet_channel[:count_channel]
                packet_channel_combination[idx] = one_packet_channel
                count_channel = len(one_packet_channel)
                temp_packet_string = \
                self.gaussian_conditional.compress(y[:, start_channel:start_channel + count_channel, :, :],
                                                   indexes[:, start_channel:start_channel + count_channel, :, :],
                                                   means=means_hat[:, start_channel:start_channel + count_channel, :,
                                                         :])[0]
            one_packet_string = b''
            one_packet_string += start_channel.to_bytes(1, byteorder='big')
            one_packet_string += int(count_channel).to_bytes(1, byteorder='big')  # count_channel
            one_packet_string += temp_packet_string
            packets.append([one_packet_string])

        print(len(packets), "True packets")
        return packets

    def compress_packets(self, x, max_packet_length, quality):
        y = self.g_a(x)
        y_copy = y.clone()
        y_chunk = y_copy.chunk(self.M // 4, dim=1)
        chanel_p = 0
        for t_y in y_chunk:
            t_swaped = swap_element(t_y)
            y[:, chanel_p:chanel_p + 4, :, :] = t_swaped
            chanel_p += 4

        z = self.h_a(y)
        if self.progressiveness:
            _, y = self.rate_less_latent(y)
            z = self.rate_less_hyper_latent(z)
            # if self.p_latent != 1: # when p_latent is very small, there will be a bug
            #     p = int(self.p_latent * y.shape[1]) + 1
            #     y = y[:, :p, :, :]

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        channel_lengths = self.get_template_channel_bytes(y, indexes, means_hat, quality, max_packet_length)
        if self.p_latent != 1:
            channel_lengths = channel_lengths[:int(self.p_latent * y.shape[1]) + 1]
        packet_channel_combination = self.get_packet_channel(channel_lengths, max_packet_length)
        packets = self.final_packets(packet_channel_combination, y, indexes, means_hat, max_packet_length)
        packets.append(z_strings)
        return {"packets": packets, "shape": z.size()[-2:]}

    def decompress_packets(self, packets, shape):
        global packet_offset
        z_hat = self.entropy_bottleneck.decompress(packets[-1], shape)
        del packets[-1]
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = torch.zeros(1, self.M , shape[0] * 4, shape[1] * 4, device=z_hat.device)
        if self.p_latent != 1:
            p = int(self.p_latent * y_hat.shape[1]) + 1

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        batch_mask = torch.zeros(y_hat.shape[0], y_hat.shape[1]).to(y_hat.device)
        if self.loss_packet:
            for i in range(len(packets)):
                one_packet = packets[i][0]
                start_channel = int.from_bytes(one_packet[0:1], byteorder='big')
                count_channel = int.from_bytes(one_packet[1:2], byteorder='big')
                one_packet = one_packet[2:]
                if gilbert_channel[packet_offset] != 1:
                    channel_data = self.gaussian_conditional.decompress([one_packet],
                                                                        indexes[:, start_channel:start_channel+count_channel , :, :],
                                                                            means=means_hat[:, start_channel:start_channel+count_channel, :, :])
                    y_hat[:, start_channel:start_channel + count_channel, :, :] = channel_data
                    batch_mask[:, start_channel:start_channel + count_channel] = 1
                else:
                    y_hat[:, start_channel:start_channel + count_channel, :, :] = 0
                    batch_mask[:, start_channel:start_channel + count_channel] = 0
                packet_offset += 1
            # print(packet_offset)
        else:
            for i in range(len(packets)):
                one_packet = packets[i][0]
                start_channel = int.from_bytes(one_packet[0:1], byteorder='big')
                count_channel = int.from_bytes(one_packet[1:2], byteorder='big')
                one_packet = one_packet[2:]
                channel_data = self.gaussian_conditional.decompress([one_packet],
                                                                    indexes[:, start_channel:start_channel+count_channel , :, :],
                                                                        means=means_hat[:, start_channel:start_channel+count_channel, :, :])
                y_hat[:, start_channel:start_channel + count_channel, :, :] = channel_data
                batch_mask[:, start_channel:start_channel + count_channel] = 1
                packet_offset += 1

        batch_mask[:, start_channel + count_channel:] = 1
        batch_mask = batch_mask.unsqueeze(2).unsqueeze(3).expand(y_hat.shape).clone()
        batch_mask_copy = batch_mask.clone()
        batch_mask_chunk = batch_mask_copy.chunk(self.M // 4, dim=1)
        chanel_p = 0
        for t_mask in batch_mask_chunk:
            t_inv_mask = inv_swap_element(t_mask)
            batch_mask[:, chanel_p:chanel_p + 4, :, :] = t_inv_mask
            chanel_p += 4

        y_hat_random_zero_copy = y_hat.clone()
        y_hat_zero_chunk = y_hat_random_zero_copy.chunk(self.M // 4, dim=1)
        chanel_p = 0
        for t_swaped in y_hat_zero_chunk:
            t_inv_swaped = inv_swap_element(t_swaped)
            y_hat[:, chanel_p:chanel_p + 4, :, :] = t_inv_swaped
            chanel_p += 4
        conv_batch_mask = self.g_conv_s(batch_mask)
        mask_and_y_hat = torch.cat((y_hat, conv_batch_mask), 1)
        mask_guidance_y = self.g_mask_s(mask_and_y_hat)
        x_hat = self.g_s(mask_guidance_y).clamp_(0, 1)
        return {"x_hat": x_hat}
def swap_element(tensor):
    tensor_return = tensor.clone()
    # tensor_return[:, 0, 0::2, 0::2] = tensor[:, 0, 0::2, 0::2]
    tensor_return[:, 0, 0::2, 1::2] = tensor[:, 1, 0::2, 0::2]
    tensor_return[:, 0, 1::2, 0::2] = tensor[:, 2, 0::2, 0::2]
    tensor_return[:, 0, 1::2, 1::2] = tensor[:, 3, 0::2, 0::2]

    tensor_return[:, 1, 0::2, 0::2] = tensor[:, 0, 0::2, 1::2]
    # tensor_return[:, 1, 0::2, 1::2] = tensor[:, 1, 0::2, 1::2]
    tensor_return[:, 1, 1::2, 0::2] = tensor[:, 2, 0::2, 1::2]
    tensor_return[:, 1, 1::2, 1::2] = tensor[:, 3, 0::2, 1::2]

    tensor_return[:, 2, 0::2, 0::2] = tensor[:, 0, 1::2, 0::2]
    tensor_return[:, 2, 0::2, 1::2] = tensor[:, 1, 1::2, 0::2]
    # tensor_return[:, 2, 1::2, 0::2] = tensor[:, 2, 1::2, 0::2]
    tensor_return[:, 2, 1::2, 1::2] = tensor[:, 3, 1::2, 0::2]

    tensor_return[:, 3, 0::2, 0::2] = tensor[:, 0, 1::2, 1::2]
    tensor_return[:, 3, 0::2, 1::2] = tensor[:, 1, 1::2, 1::2]
    tensor_return[:, 3, 1::2, 0::2] = tensor[:, 2, 1::2, 1::2]
    # tensor_return[:, 3, 1::2, 1::2] = tensor[:, 3, 1::2, 1::2]
    return tensor_return

def inv_swap_element(tensor_swaped):
    tensor_return = tensor_swaped.clone()
    # tensor_return[:, 0, 0::2, 0::2] = tensor_swaped[:, 0, 0::2, 0::2]
    tensor_return[:, 1, 0::2, 0::2] = tensor_swaped[:, 0, 0::2, 1::2]
    tensor_return[:, 2, 0::2, 0::2] = tensor_swaped[:, 0, 1::2, 0::2]
    tensor_return[:, 3, 0::2, 0::2] = tensor_swaped[:, 0, 1::2, 1::2]

    tensor_return[:, 0, 0::2, 1::2] = tensor_swaped[:, 1, 0::2, 0::2]
    # tensor_return[:, 1, 0::2, 1::2] = tensor_swaped[:, 1, 0::2, 1::2]
    tensor_return[:, 2, 0::2, 1::2] = tensor_swaped[:, 1, 1::2, 0::2]
    tensor_return[:, 3, 0::2, 1::2] = tensor_swaped[:, 1, 1::2, 1::2]

    tensor_return[:, 0, 1::2, 0::2] = tensor_swaped[:, 2, 0::2, 0::2]
    tensor_return[:, 1, 1::2, 0::2] = tensor_swaped[:, 2, 0::2, 1::2]
    # tensor_return[:, 2, 1::2, 0::2] = tensor_swaped[:, 2, 1::2, 0::2]
    tensor_return[:, 3, 1::2, 0::2] = tensor_swaped[:, 2, 1::2, 1::2]

    tensor_return[:, 0, 1::2, 1::2] = tensor_swaped[:, 3, 0::2, 0::2]
    tensor_return[:, 1, 1::2, 1::2] = tensor_swaped[:, 3, 0::2, 1::2]
    tensor_return[:, 2, 1::2, 1::2] = tensor_swaped[:, 3, 1::2, 0::2]
    # tensor_return[:, 3, 1::2, 1::2] = tensor_swaped[:, 3, 1::2, 1::2]
    return tensor_return




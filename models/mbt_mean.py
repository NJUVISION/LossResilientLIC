import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.registry import register_model
import numpy as np
import random
import math

from .base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from .utils import conv, deconv
from utils import get_config

dataset_total_packet = 0
batch_test_config_path = "codec_config.yaml"
config = get_config(batch_test_config_path )
# print(f"config from {batch_test_config_path} ", __file__)
with open(config["channel"], 'r') as f:
    gilbert_channel = f.read()
    gilbert_channel = eval(gilbert_channel)
packet_offset = 0
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


@register_model("bmshj2018-hyperprior")
class ScaleHyperprior(CompressionModel):
    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )


        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
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
class MeanScaleHyperprior(ScaleHyperprior):
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

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
    def set_progressive_level(self, progressiveness,  p_hyper, p):
        self.progressiveness = progressiveness
        self.p_hyper_latent = p_hyper
        self.p_latent = p



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

        return temp_data

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


    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        if self.progressiveness:
            y_likelihoods = self.drop_zeros_likelihood(y_likelihoods, y)
            z_likelihoods = self.drop_zeros_likelihood(z_likelihoods, z)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        # for i in range(y.shape[1]):
        #     print(len(self.gaussian_conditional.compress(y[:, i, :, :], indexes[:, i, :, :], means=means_hat[:, i, :, :])[0]))

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, drop_channel_index):
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
        print(drop_channel_index)
        y_hat_p[:, drop_channel_index, :, :] = 0
        if self.p_latent != 1:
            y_hat[:, :p, :, :] = y_hat_p
        else:
            y_hat = y_hat_p
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


    def get_template_channel_bytes(self, y, index, means, q, max_packet_length):
        y_fit = []
        for i in range(y.shape[1]):
            ith_channel_string_length = len(self.gaussian_conditional.compress(y[:, i, :, :], index[:, i, :, :], means=means[:, i, :, :])[0])
            # print(ith_channel_string_length)
            # assert ith_channel_string_length <= max_packet_length - 4
            y_fit.append(ith_channel_string_length)
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

    def final_packets(self, packet_channel_combination, y, indexes, means_hat, max_packet_length):
        packets = []
        for idx, one_packet_channel in enumerate(packet_channel_combination):
            start_channel = one_packet_channel[0]
            count_channel = len(one_packet_channel)

            temp_packet_string = self.gaussian_conditional.compress(y[:, start_channel:start_channel + count_channel, :, :],
                                               indexes[:, start_channel:start_channel + count_channel, :, :],
                                               means=means_hat[:, start_channel:start_channel + count_channel, :, :])[0]
            while (len(temp_packet_string) > max_packet_length - 4):
                count_channel = math.ceil(count_channel / 2)
                packet_channel_combination.insert(idx + 1, one_packet_channel[count_channel:])
                one_packet_channel = one_packet_channel[:count_channel]
                packet_channel_combination[idx] = one_packet_channel
                count_channel = len(one_packet_channel)
                temp_packet_string = self.gaussian_conditional.compress(y[:, start_channel:start_channel + count_channel, :, :],
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
        z = self.h_a(y)
        print(f"keeping {int(self.p_latent * y.shape[1]) + 1} channel")

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        channel_lengths = self.get_template_channel_bytes(y, indexes, means_hat, quality, max_packet_length)
        packet_channel_combination = self.get_packet_channel(channel_lengths, max_packet_length)
        # 正式分包
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

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
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
                else:
                    y_hat[:, start_channel:start_channel + count_channel, :, :] = 0
                packet_offset += 1
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
                packet_offset += 1
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



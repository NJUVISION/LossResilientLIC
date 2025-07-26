import os
import yaml
from utils import get_config

def updata_yaml(input_path, output_path,  model, p, channel):
    config = get_config(input_path)
    config['model'] = model
    config['p_latent'] = p
    config['p_hyper'] = p
    config['channel'] = channel
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config,f)

channel = "./exp/gilbert_channel/0/channel0.txt"
model = "LRLICWProg"
p_list = [i / 100 * 4 + 0.04 for i in range(15)]
config = get_config("codec_config.yaml")
config["model"] = model
config["channel"] = channel
bpps = []
psnrs = []


for p in p_list:
    updata_yaml("codec_config.yaml", "codec_config.yaml", model, p, channel)
    os.system("python eval_channel_packet.py codec_config.yaml")
    with open("log.txt", 'r') as f:
        mean_line = f.readlines()[-1]
    mean_line_list = mean_line.split("\t")
    assert mean_line_list[0] == "meanbppandpsnr"
    mean_bpp = mean_line_list[1]
    mean_psnr = mean_line_list[2]
    bpps.append(mean_bpp)
    psnrs.append(mean_psnr)
    print(f"p={p}\t{mean_bpp}\t{mean_psnr}")
with open("proglog.txt", 'w') as f:
    for i in range(len(p_list)):
        f.write(str(p_list[i]) + '\t' + str(bpps[i]) + '\t' + str(psnrs[i])[:-1] + '\n')
dataset_total_packet = 56800
channel = []
out_path = "./gilbert_channel/0/channel0.txt"
for i in range(dataset_total_packet):
    channel.append(0)
with open(out_path, 'w') as f:
    f.write(str(channel))
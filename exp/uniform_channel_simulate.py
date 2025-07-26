import random


target_pe = [20] # target packet loss rate in percentage

dataset_total_packet = 56800
channel = 10 # number of channels for each pe


def generate_txt(pe, pe_str, time):
    # 1 for packet loss, 0 for packet received
    uniform_channel = [1 if random.random() < pe else 0 for _ in range(dataset_total_packet)]
    out_path = './uniform_channel/' + pe_str + '/channel' + str(time) + '.txt'
    with open(out_path, "w") as f:
        f.write(str(uniform_channel))


for i in range(len(target_pe)):
    for j in range(channel):
        generate_txt(target_pe[i] / 100, str(target_pe[i]), j)


# check Pe
with open("./uniform_channel/20/channel3.txt", 'r') as f:
    gilbert_channel = f.read()
    gilbert_channel = eval(gilbert_channel)
cout_0 = 0
cout_1 = 0
for i in gilbert_channel[:600]:
    if i == 0:
        cout_0 += 1
    else:
        cout_1 += 1
print("0:", cout_0 / (cout_0 + cout_1))
print("1:", cout_1 / (cout_0 + cout_1))

from sim2net.packet_loss.gilbert_elliott import GilbertElliott

prhks = [
[ 0.417288193847740, 0.973672452311393, 0.620000000000000, 0.948571428571429],
]

target_pr = [15,] # target packet loss rate in percentage

dataset_total_packet = 56800
channel = 10 # number of channels for each prhk

def generate_txt(prhk, pe, time):
    ge = GilbertElliott(prhk)
    gilbert_channel = []
    for i in range(dataset_total_packet):
        pl = int(ge.packet_loss())
        gilbert_channel.append(pl)
    out_path = './gilbert_channel/' + pe + '/channel' + str(time) + '.txt'
    with open(out_path, 'w') as f:
        f.write(str(gilbert_channel))



for i in range(len(prhks)):
    pe = ((1-prhks[i][3]) * prhks[i][1] + (1 - prhks[i][2]) * prhks[i][0]) / (prhks[i][0] + prhks[i][1])
    print("Pe: ", pe)
    for j in range(channel):
        generate_txt(prhks[i], str(target_pr[i]), j)

# check Pe
with open("./gilbert_channel/10/channel2.txt", 'r') as f:
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

from lxml import etree
import torch
from os import listdir
from os.path import join
save_file = 'data.json'
path = '1k_val'
name = '1k_val'
data = []
set = []
size = 256
with open(save_file, "w") as outfile:
    for file in listdir(path):
        f = (join(path,file))
        tree = etree.parse((f))
        root = tree.getroot()
        strokes = []
        for i in root:
            temp = []
            d = etree.tostring(i)
            d = d.decode(encoding='utf_8')
            data = d.split()

            temp.extend([float(data[3][0:-2]) / size, float(data[4][0:-2]) / size, float(data[6][0:-2]) / size,
                         float(data[7][0:-2]) / size, float(data[8][0:-2]) / size, float(data[9][0:-2]) / size])
            # float(data[16][14:-3]) / self.size])
            for i in range(len(temp)):
                temp[i] = (temp[i] * 2) - 1
            strokes.append(temp)
        strokes = torch.FloatTensor(strokes)
        set.append(strokes)

    torch.save(set, 'Data/{}.pt'.format(name))

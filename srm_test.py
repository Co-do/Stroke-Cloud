from Data_Set  import  my_collate, Tensor
from models import  srm
from torch.utils.data import DataLoader
import os
import torch
from utils import sample, draw


experiment_name = 'SRM Test'
torch.set_float32_matmul_precision('medium')
test_path = "./Data/10k.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = srm.load_from_checkpoint("./Models/SRM.ckpt")
size = 512
dim_in = 6
samples = 1000
L = []
#To help train the lsg we create our latent data by sampling the data set multiple times.
reps = 1
if not os.path.exists("Samples/{}".format(experiment_name)):
        os.makedirs("Samples/{}".format(experiment_name))

test_set = Tensor(test_path)
loader = DataLoader(test_set, 1, shuffle=False, collate_fn= my_collate, pin_memory=True)
Encoder = model.encoder
Decoder = model.decoder

for i in range(reps):
    with torch.no_grad():
        for i, data in enumerate(loader):
            Latent = Encoder(data[0].to(device))[0]
            L.append(Latent)
            stroke = sample(samples, model.sample_steps, Decoder, model.noise_scheduler_sample, Latent, dim_in)
            filename = 'Samples/{}/{}.svg'.format(experiment_name, i)
            draw(model.format, size, filename, stroke)


Latents = [item for sublist in L for item in sublist]
torch.save(Latents, 'Latent/{}.pt'.format(experiment_name))



import wandb
from Data_Set  import  my_collate, Tensor
from models import SetTransformer, MLP, srm
import torch
from torch.utils.data import DataLoader
import os
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from diffusers import DDIMScheduler, DDPMScheduler
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
experiment_name = 'First Run'
format_path = 'format.svg'
train_path = 'Data/10k.pt'
val_path = 'Data/10k_val.pt'


learning_rate = 2e-4
size = 512
BATCH_SIZE = 128
hidden_size = 4096
samples = 1000
steps = 200
sample_steps = 30
beta_schedule = 'linear'
dim_in = 6
gpu_num = 1

#Add WB key here
wand_b_key = 'Your key here'
wandb.login(key=wand_b_key)
wandb_logger = WandbLogger(name=experiment_name,project='Your Stroke Cloud')
trainer = Trainer(logger=wandb_logger)
train_set = Tensor(train_path)
val_set = Tensor(val_path)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, collate_fn= my_collate, pin_memory=True)
val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, collate_fn= my_collate)
torch.set_float32_matmul_precision("medium")

checkpoint_callback = ModelCheckpoint(
    dirpath="Models/{}/".format(experiment_name),
    filename="{epoch:02d}-{global_step}",
)

decoder = MLP(
        hidden_size=hidden_size,
        hidden_layers=6,
        emb_size=64,
        time_emb= "sinusoidal",
        input_emb = "sinusoidal")

encoder = SetTransformer(
        dim_input=dim_in,
        num_outputs=1,
        dim_output=256,
        num_inds=32,
        dim_hidden=256,
        num_heads=16,
        ln=True)

if not os.path.exists("Results/{}".format(experiment_name)):
        os.makedirs("Results/{}".format(experiment_name))

if not os.path.exists("Models/{}".format(experiment_name)):
        os.makedirs("Models/{}".format(experiment_name))

scheduler = DDPMScheduler(beta_end=1e-4, beta_start=1e-5, num_train_timesteps = steps, beta_schedule=beta_schedule)
ddim_s = DDIMScheduler(beta_end=1e-4, beta_start=1e-5, num_train_timesteps = steps, beta_schedule=beta_schedule)
ddim_s.set_timesteps(sample_steps)
sample_steps = list(range(sample_steps))
srm = srm(encoder, decoder, scheduler, ddim_s, experiment_name, samples, sample_steps, format_path, size,dim_in, learning_rate)


trainer = L.Trainer(accelerator='gpu', devices=gpu_num, strategy='auto' ,logger=wandb_logger, max_epochs= -1,
                    check_val_every_n_epoch=100, enable_progress_bar=True, profiler="simple",
                    callbacks=[StochasticWeightAveraging(swa_lrs=learning_rate),checkpoint_callback ], benchmark=True)
trainer.fit(model=srm, train_dataloaders=train_loader, val_dataloaders= val_loader)

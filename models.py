
from torch import optim
import lightning as L
from positional_embeddings import PositionalEmbedding
from utils import sample, draw, l_sample
from modules import *
device = "cuda" if torch.cuda.is_available() else "cpu"

class l_Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor):
        return x + self.act(self.norm(self.ff(x)))

class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal",):
        super(MLP, self).__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp3 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp4 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp5 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp6 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        concat_size = (7 * emb_size) + 256
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(l_Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 6))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, y):

        t = t.to(device)
        x1_emb = self.input_mlp1(x[:, :, 0])
        x2_emb = self.input_mlp2(x[:, :, 1])
        x3_emb = self.input_mlp3(x[:, :, 2])
        x4_emb = self.input_mlp4(x[:, :, 3])
        x5_emb = self.input_mlp5(x[:, :, 4])
        x6_emb = self.input_mlp6(x[:, :, 5])
        t_emb = self.time_mlp(t)
        t_emb = t_emb.repeat(x1_emb.shape[0], 1, 1)
        y = y.repeat(1,x1_emb.shape[1],1 )
        x = torch.cat((x1_emb, x2_emb, x3_emb, x4_emb, x5_emb, x6_emb, t_emb, y), dim=-1)
        x = self.joint_mlp(x)

        return x
class srm(L.LightningModule):
    def __init__(self, encoder, decoder, noise_scheduler,noise_scheduler_sample, experiment_name, samples, sample_steps, format_path, sample_size,dim_in, lr):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_sample = noise_scheduler_sample
        self.experiment_name = experiment_name
        self.samples = samples
        self.sample_size = sample_size
        self.sample_steps = sample_steps
        self.format = format_path
        self.learning_rate = lr
        self.dim_in = dim_in
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, ):
        #Encoder
        Set = batch[0]
        condition, mu, sigma = self.encoder(Set)

        #Decoder
        #1 instead of 0 to use collate
        Strokes = batch[1]
        noise = torch.randn(Strokes.shape, device=self.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (Strokes.shape[1],),device=self.device).long()
        noisy = self.noise_scheduler.add_noise(Strokes, noise, timesteps)

        #Train
        noise_pred = self.decoder(noisy, timesteps, condition)
        loss_mse = F.mse_loss(noise_pred, noise,)

        #KL
        KLD = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        KLS = (3e-8) #* self.current_epoch
        loss = loss_mse + (KLS * KLD)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):

        condition, mu, sigma = self.encoder(batch[0])
        for i in range(len(condition)):
            filename = 'Results/{}/{}_{}.svg'.format(self.experiment_name, self.current_epoch, i)
            stroke = sample(self.samples, self.sample_steps, self.decoder, self.noise_scheduler_sample, mu[i], self.dim_in)
            draw(self.format, self.sample_size, filename, stroke)

    def test_step(self, batch, batch_idx):

        condition, mu, sigma = self.encoder(batch[0])
        filename = 'Samples/{}/{}.svg'.format(self.experiment_name, batch_idx)
        stroke = sample(self.samples, self.sample_steps, self.decoder, self.noise_scheduler_sample, mu, self.dim_in)
        draw(self.format, self.sample_size, filename, stroke)

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=(self.learning_rate))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000], gamma=0.5)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

class lsg(L.LightningModule):
    def __init__(self, model, srm, experiment_name, timesteps, noise_scheduler,noise_scheduler_sample, learning_rate):
        super().__init__()
        self.model = model
        self.srm = srm
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_sample = noise_scheduler_sample
        self.learning_rate = learning_rate
        self.save_hyperparameters()



    def training_step(self, batch, batch_idx, ):

        #Encoder
        latent = batch
        noise = torch.randn(latent.shape, device=self.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (latent.shape[1],),device=self.device).long()
        noisy = self.noise_scheduler.add_noise(latent, noise, timesteps)
        noise_pred = self.model(noisy, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        sch = self.lr_schedulers()
        #sch.step()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):

        Latent = l_sample(self.timesteps, self.model, self.noise_scheduler_sample)
        stroke = sample(self.srm.samples, self.srm.sample_steps, self.srm.decoder, self.srm.noise_scheduler_sample, Latent, self.srm.dim_in)
        filename = 'Results/{}/{}.svg'.format(self.experiment_name, self.current_epoch)
        draw(self.srm.format, self.srm.sample_size, filename, stroke)


    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=(self.learning_rate))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 1000000, 2000000], gamma=0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class L_MLP(nn.Module):
    def __init__(self, hidden_size: int = 1024, hidden_layers: int =6, emb_size: int =64,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal",):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)

        concat_size = emb_size + 256

        layers = [nn.Linear(concat_size, hidden_size),nn.LayerNorm(hidden_size), nn.GELU()]

        for _ in range(hidden_layers):
            layers.append(l_Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 256))
        self.joint_mlp = nn.Sequential(*layers)


    def forward(self, x, t):

        t_emb = self.time_mlp(t)
        t_emb = t_emb.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, t_emb), dim=-1)
        x = self.joint_mlp(x)

        return x

class SetTransformer(L.LightningModule):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds, dim_hidden, num_heads, ln):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                )
        self.linear_mu = nn.Linear(dim_hidden, dim_output)
        self.linear_sigma = nn.Linear(dim_hidden, dim_output)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device)  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(self.device)


    def forward(self, X):

         y = self.dec(self.enc(X))
         mu = self.linear_mu(y)
         sigma = torch.exp(self.linear_sigma(y))
         z = mu + sigma * self.N.sample(mu.shape).to(self.device)
         return z, mu, sigma



class NoiseScheduler(L.LightningModule):
    def __init__(self,num_timesteps=1000, beta_start=0.0001,beta_end=0.02, beta_schedule="linear"):
        super(NoiseScheduler, self).__init__()
        #self.device = device
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32, device=self.device)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0, dtype=torch.float32)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = (self.alphas_cumprod ** 0.5)
        self.sqrt_one_minus_alphas_cumprod = ((1 - self.alphas_cumprod) ** 0.5)

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):

        s1 = self.sqrt_inv_alphas_cumprod[t].to(self.device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].to(self.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t].to(self.device)
        s2 = self.posterior_mean_coef2[t].to(self.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t]).to(self.device)
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep.to(self.device)
        pred_original_sample = self.reconstruct_x0(sample, t, model_output).to(self.device)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t).to(self.device)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample.to(self.device)

    def add_noise(self, x_start, x_noise, timesteps):


        s1 = self.sqrt_alphas_cumprod.to(self.device)[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod.to(self.device)[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return (s1 * x_start + s2 * x_noise)

    def __len__(self):
        return self.num_timesteps

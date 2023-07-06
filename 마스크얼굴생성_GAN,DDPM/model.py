import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import math
from modules import *

from torchvision.utils import save_image
import DDPM_fid as fid

class DiffusionModel(pl.LightningModule):
    def __init__(self, in_size, t_range, img_depth,n_class=2,light=False):
        super().__init__()
        self.time_dim = 256
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size
        self.label_emb = nn.Embedding(n_class,self.time_dim) 

        bilinear = True
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(img_depth, 64)        # channel : 1 -> 64
        self.down1 = Down(64, 128)                  # channel : 64 -> 128
        self.down2 = Down(128, 256)                 # channel : 128 -> 256
        self.down3 = Down(256, 512 // factor)       # channel : 256 -> 256

        self.up1 = Up(512, 256 // factor) # channel : 512 -> 128
        self.up2 = Up(256, 128 // factor) # channel : 256 -> 256
        self.up3 = Up(128, 64)            # channel : 256 -> 256

        self.sa1 = SelfAttention(128)
        self.sa2 = SelfAttention(256)
        self.sa3 = SelfAttention(256)
        self.sa4 = SelfAttention(128)
        self.sa5 = SelfAttention(64)
        self.sa6 = SelfAttention(64)

        self.remove_deep_conv = light
        if self.remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)

        self.outc = nn.Conv2d(64, img_depth, kernel_size=1)

        self.training_step_outputs = []
        self.epoch = 0

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, c=None):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        t = self.pos_encoding(t, self.time_dim)

        if c is not None:
            t += self.label_emb(c)    
                                   
        x1 = self.inc(x)                                        # 1,64,64 -> 64,64,64
        x2 = self.down1(x1,t)                                   # 128,32,32
        x2 = self.sa1(x2)
        x3 = self.down2(x2,t)                                   # 256,16,16
        x3 = self.sa2(x3)                                       # 256,16,16
        x4 = self.down3(x3,t)                                   # 256,8,8
        x4 = self.sa3(x4)                                       # 256,8,8
        
        x4 = self.bot1(x4)                                      # 256,8,8
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)                                  # 512,8,8
        x4 = self.bot3(x4)                                      # 256,8,8
        
        x = self.up1(x4, x3 , t)                                # 128,16,16 + 128,16,16
        x = self.sa4(x)                                         # 128,16,16
        x = self.up2(x, x2, t) 
        x = self.sa5(x)                                         # 64,32,32 + 64,32,32
        x = self.up3(x, x1, t)                                  # 64,64,64 + 64,64,64
        x = self.sa6(x)
        output = self.outc(x)                                   # 64,64,64 -> 1,64,64
        return output

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small
        )

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        batch , label = batch
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float), label)
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def denoise_sample(self, x, t, c=None):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape,device=self.device)
            else:
                z = 0
            if c is not None:
                e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1),c)
            else:
                e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1))
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.training_step_outputs.append(loss)
        self.log("train/loss", loss)
        return loss

    def on_train_epoch_end(self):
        opt = self.optimizers()
        for param_group in opt.param_groups: 
            print(f"{self.current_epoch}epoch lr : {param_group['lr']}")
            
        loss = torch.stack(self.training_step_outputs).mean()
        self.training_step_outputs.clear()  # free memory
        print('train/loss : {:.5f}'.format(loss))

        if self.epoch >= 100 :
            # 마스크 이미지 생성
            x = torch.randn((4, 1, 64, 64)).cuda()
            generated_labels = torch.cuda.IntTensor(4).fill_(0)
            sample_steps = torch.arange(self.t_range-1, 0, -1).cuda()
            for t in sample_steps:
                x = self.denoise_sample(x, t,generated_labels)
            for j in range(4):
                save_image(x[j], f'./results/DDPM/with_mask_val/{j}.png', normalize=True)
            print(f'val - w/ mask : {fid.fid_return("./results/DDPM/with_mask_val",4,True)}')
            
            # 마스크 없는 이미지 생성
            x = torch.randn((4, 1, 64, 64)).cuda()
            generated_labels = torch.cuda.IntTensor(4).fill_(1)
            sample_steps = torch.arange(self.t_range-1, 0, -1).cuda()
            for t in sample_steps:
                x = self.denoise_sample(x, t,generated_labels)
            for j in range(4):
                save_image(x[j], f'./results/DDPM/without_mask_val/{j}.png', normalize=True)
            print(f'val - w/o mask : {fid.fid_return("./results/DDPM/without_mask_val",4,False)}')
        self.epoch += 1
        print()

        


    def validation_step(self, batch, batch_idx):
        #loss = self.get_loss(batch, batch_idx)
        #self.log("val/loss", loss)
        return

    def on_validation_epoch_end(self, outputs):
      # called at the end of the validation epoch
      # outputs is an array with what you returned in validation_step for each batch
      # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 

      #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
      #tensorboard_logs = {'val_loss': avg_loss}
      #return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
      print(f'validation/FID : 0')
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch if epoch <= 10 else 0.95 ** epoch)
        return [optimizer] , [scheduler]

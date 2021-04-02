import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import pandas
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils

##构建残差块  针对与鉴别器
#residual block
class D_Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
           

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def D_resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(D_Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(D_Residual(out_channels, out_channels))
    return nn.Sequential(*blk)
 
 ##构建残差块  针对与生成器
#residual block
class G_Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
           

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def G_resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(G_Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(G_Residual(out_channels, out_channels))
    return nn.Sequential(*blk)
    
 """
初始化权重
"""
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

##构建鉴别器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(3, 64, 4, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
           
           
            D_resnet_block(128, 128, 2, first_block=True),
            
            D_resnet_block(128, 256, 2),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
           
            nn.Conv2d(512, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
        
        
        #构建生成器
# 生成器代码
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 输入是Z，进入卷积
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            G_resnet_block(256, 128, 2),
            G_resnet_block(128, 128, 2,first_block=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.model(input)
        
        
 class CelebaDataset(Dataset):

    def __init__(self,root_dir,label_dir,transforms):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.dataset = os.listdir(os.path.join( self.root_dir,self.label_dir))[0:10000]
        self.transforms = transforms

        pass

    def __getitem__(self, index):
        image_name = self.dataset[index]
        image_item = os.path.join(self.root_dir,self.label_dir,image_name)
        img = Image.open(image_item).convert('RGB')
        return self.transforms(img)
        # 128 * 128 * 3
        #return  torch.cuda.FloatTensor(img).permute(2,0,1).view(1,3,128,128) / 255.0
        #return img
        

    def __len__(self):
        return len(self.dataset)

    def plot_image(self, index):
        img = Image.open(os.path.join(self.root_dir, self.label_dir,self.dataset[index])).convert('RGB')
        img = self.transforms(img) 
       
        plt.imshow(img.permute(1,2,0), interpolation='nearest')
        pass

#图片大小
image_size = 64
transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
#加载数据集
celeba_dataset = CelebaDataset('../input/celeba-dataset/img_align_celeba','img_align_celeba',transforms = transform)
#测试图片
celeba_dataset.plot_image(0)


"""
Training of WGAN-GP
"""


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
CHANNELS_IMG = 1
Z_DIM = 100
num_epochs = 10
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
lr = 0.0002
beta1 = 0.5
dataloader = DataLoader(
    celeba_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

#--------------------------------train
# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])


gen = Generator().to(device)
critic = Discriminator().to(device)
initialize_weights(gen)
initialize_weights(critic)

# 初始化BCELoss函数
criterion = nn.BCELoss()

# 创建一批潜在的向量，我们将用它来可视化生成器的进程
fixed_noise = torch.randn(128, 100, 1, 1, device=device)

# 在训练期间建立真假标签的惯例
real_label = 1
fake_label = 0

# 为 G 和 D 设置 Adam 优化器
optimizerD = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

#-------------------------------------------------------------------
print("开始训练---")
# For each epoch
for epoch in range(num_epochs):
    # 对于数据加载器中的每个batch
    for i, data in enumerate(dataloader, 0):
        
        #data 128 3 128 128
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## 训练所有的真实数据
        critic.zero_grad()
        # 取一个batch
        real_cpu = data.to(device)
        #print("real_cpu:")
        #print(real_cpu.shape)
        #print(real_cpu.shape) 128 * 3 *  128 * 128
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device).float()
        #print(label) ##[1,1,1,1]
        # 鉴别器输出
        # errD_real = criterion(output, label)
        
        output = critic(real_cpu).view(-1)
        #output = critic(real_cpu)
        
        # 计算真实图像的损失
        errD_real = criterion(output, label)
        # 梯度下降，自动求导
        errD_real.backward()
        D_x = output.mean().item()

        ## 训练所有的假图片
        # 生成随机噪声
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        # 使用生成器生成      
        fake = gen(noise)
        label.fill_(fake_label)
        # 将假的图片送到鉴别器
        output = critic(fake.detach()).view(-1)
        # 计算假图片的损失值
        errD_fake = criterion(output, label)
        # 梯度下降
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 计算损失值
        errD = errD_real + errD_fake
        # Update D
       
        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        gen.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = critic(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        #print(errG)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
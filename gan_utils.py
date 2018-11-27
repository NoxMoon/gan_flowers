import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from skimage import io, transform
from tqdm import tqdm

class flower_dataset(Dataset):
    
    def __init__(self, image_paths, transform=None):
        self.len = len(image_paths)
        self.transform = transform
        self.X = []
        for path in tqdm(image_paths):
            image = io.imread(path)
            if self.transform is not None:
                image = transform(image)
            self.X.append(image)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.X[idx]

class flower_mask_dataset(Dataset):
    
    def __init__(self, image_paths, mask_paths=None, mask_sizes=None, 
                 transform=None, mask_transform=None, in_memory=True):
        self.len = len(image_paths)
        self.transform = transform
        self.mask_transform = mask_transform
        self.in_memory = in_memory
        self.mask_sizes = mask_sizes
        
        if not self.in_memory:
            self.image_paths = image_paths
            if mask_paths is None:
                self.mask_paths = [None]*self.len
            else:
                self.mask_paths = mask_paths
            return
        
        self.image = []
        self.mask = []
        
        for i, path in tqdm(enumerate(image_paths)):
            image = io.imread(path)
            if (len(image.shape)!=3):
                self.len -= 1
                continue
                
            if self.transform is not None:
                image = self.transform(image)
            self.image.append(image)

            if mask_paths is None:
                self.mask.append(self.get_empty_masks())
            else:
                mask = io.imread(mask_paths[i], as_gray=True)
                masks = []
                for size in self.mask_sizes:
                    masks.append(self.get_mask(mask, size))
                self.mask.append(masks)

        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.in_memory:
            return self.image[idx], self.mask[idx]
        else:
            image_path = self.image_paths[idx]
            image = io.imread(image_path)
            if self.transform is not None:
                image = self.transform(image)
            
            mask_path = self.mask_paths[idx]
            if mask_path is None:
                masks = self.get_empty_masks()
            else:
                mask = io.imread(mask_path)
                masks = []
                for size in self.mask_sizes:
                    masks.append(self.get_mask(mask, size))
            return image, masks
            
    def get_empty_masks(self):
            return [-torch.ones(1,size,size,dtype=torch.float32) for size in self.mask_sizes]
        
    def get_mask(self, mask_, size=None):
        mask = mask_
        if (size is not None):
            mask = transform.resize(mask, (size, size)).astype('float32')
        if self.mask_transform is not None:
            mask = self.mask_transform(np.expand_dims(mask,axis=-1))
        return mask
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def sample_mask(batch_size, sizes, realdata, noise_level=0.1, device=None):
    masks = [torch.randn(batch_size, 1, size, size, device=device, dtype=torch.float32) * noise_level for size in sizes]
    for i in range(batch_size):
        n = np.random.randint(len(realdata))
        for k, mask in enumerate(masks):
            mask[i,:,:,:] += realdata[n][1][k].to(device)
    return masks

def mask_pair_to_label(mask, real_mask, device=None):
    mask1 = mask[-1]
    mask2 = real_mask[-1]
    b_size = mask1.size()[0]
    mask_size = mask1.size()[-1]**2
    #label = ((mask1-mask2).abs()<0.4).view(b_size,-1).sum(1).float()/mask_size
    #label = label.to(device)
    label = torch.ones(b_size, device=device)
    label = label - ((mask1-mask2).abs()/2.0).view(b_size,-1).mean(1)
    return label

def get_label(size, real, soft=0.2, noise=False, noise_level=0.1, device=None):
    label = torch.full((size,), real, device=device)
    if real == 1:
        label -= torch.rand(size, device=device) * soft
    else:
        label += torch.rand(size, device=device) * soft
    if noise:
        perm = torch.randperm(label.size(0))
        idx = perm[:int(label.size(0)*noise_level)]
        label[idx] = 1 - label[idx]
    return label

def noise_input(data, noise_level=0, device=None, clip=False):
    if noise_level==0:
        return data.to(device)
    else:
        noise_data = torch.randn(data.size(), device=device) * noise_level
        noise_data += data.to(device)
        if clip:
            noise_data = torch.clamp(noise_data, -1, 1)
    return noise_data

def sigmoid(x, alpha=0.01):
    return 1/(1+np.exp(-alpha*x))

def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
        
class Generator(nn.Module):
    def __init__(self, ngpu, ngf, nz, nc, mask_sizes=[]):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.nz = nz
        self.nc = nc
        self.mask_sizes = mask_sizes
        self.m2 = int(2 in mask_sizes)
        self.m4 = int(4 in mask_sizes)
        self.m8 = int(8 in mask_sizes)
        self.m16 = int(16 in mask_sizes)
        self.m32 = int(32 in mask_sizes)
        self.layer1 = nn.Sequential(
            # input is Z, going into a convolution
            # input Z(32,2,2)
            nn.ConvTranspose2d(self.nz + self.m2, self.ngf * 8, 4, 2, 0, bias=False),
            nn.ReflectionPad2d(-1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8 + self.m4, self.ngf * 4, 4, 2, 0, bias=False),
            nn.ReflectionPad2d(-1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            # state size. (ngf*4) x 8 x 8            
            nn.ConvTranspose2d(self.ngf * 4 + self.m8, self.ngf * 2, 4, 2, 0, bias=False),
            nn.ReflectionPad2d(-1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2 + self.m16, self.ngf, 4, 2, 0, bias=False),
            nn.ReflectionPad2d(-1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.layer5 = nn.Sequential(
            # state size. (ngf) x 32 x 32            
            nn.ConvTranspose2d(self.ngf + self.m32, self.nc, 4, 2, 0, bias=False),
            nn.ReflectionPad2d(-1),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise, mask=None):
        if self.m2 == 1:
            x = torch.cat((noise, mask[0]), dim=1) #2*2
            x = self.layer1(x)
        else:
            x = self.layer1(noise)
            
        if self.m4 == 1:
            x = torch.cat((x, mask[1]), dim=1) #4*4
        x = self.layer2(x)
        
        if self.m8 == 1:
            x = torch.cat((x, mask[2]), dim=1) #8*8
        x = self.layer3(x)
        
        if self.m16 == 1:
            x = torch.cat((x, mask[3]), dim=1) #16*16
        x = self.layer4(x)
        
        if self.m32 == 1:
            x = torch.cat((x, mask[4]), dim=1) #32*32
        x = self.layer5(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc, mask_sizes=[]):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.nc = nc
        self.mask_sizes = mask_sizes
        self.m4 = int(4 in mask_sizes)
        self.m8 = int(8 in mask_sizes)
        self.m16 = int(16 in mask_sizes)
        self.m32 = int(32 in mask_sizes)
        self.layer1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf + self.m32, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2 + self.m16, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4 + self.m8, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
        # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8 + self.m4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image, mask=None):
        x = self.layer1(image)
        
        if self.m32 == 1:
            x = torch.cat((x, mask[4]), dim=1) #32*32
        x = self.layer2(x)
        
        if self.m16 == 1:
            x = torch.cat((x, mask[3]), dim=1) #16*16
        x = self.layer3(x)
        
        if self.m8 == 1:
            x = torch.cat((x, mask[2]), dim=1) #8*8
        x = self.layer4(x)
        
        if self.m4 == 1:
            x = torch.cat((x, mask[1]), dim=1) #4*4
        x = self.layer5(x)
        return x
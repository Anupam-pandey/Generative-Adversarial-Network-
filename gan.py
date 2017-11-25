'''
Team : K-NN Classifiers
1. Adithya Avvaru
2. Gaurav Agarwal
3. Anupam Pandey
4. Marimganti Srinivas

'''
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as dset
import torch.optim as optim
import torchvision.utils as utils
import argparse
import os
import random
from torch.autograd import Variable

#Default Values for: BatchSize, Workers, No. of Filters in Generator and Discriminator
batchSize = 100 # Number of Images to be taken at a single time
workers = 4 # Number of cores to be used for Data Loading Purpose
ngf = 64    # Number of filters in Generator
ndf = 64    # Number of filters in Discriminator
nc = 3      # Number of Channels in input Image
nz = 100    # Dimension of Noise Space 
iterations = 25 # No of iterations
imageSize = 32  # Image Size 32*32 -- for CIFAR dataset
learning_rate = 0.0002 # Learning rate

# Parsing Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='Path to Dataset')
parser.add_argument('--out_folder', default='.', help='Folder to Output Images')

options = parser.parse_args()
print("Given Options are : ")
print(options)

try:
    os.makedirs(options.out_folder)
except OSError:
    pass

# Image Preprocessing
transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

dataset = dset.CIFAR10(root=options.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                         shuffle=True, num_workers=workers)

       
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Input : BatchSize * NoiseSize * 1 * 1
        # Input : 100 * 100 * 1 * 1
        self.main = nn.Sequential(
# Usage:    nn.ConvTranspose2d(inputChannels, outputChannels, KernelSize, Stride, Padding, bias=False)
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            # New State : 100 * 512 * 4 * 4
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            # New State : 100 * 256 * 8 * 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            # New State : 100 * 128 * 16 * 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
            # New State : 100 * 3 * 32 * 32
            nn.BatchNorm2d(nc),
            nn.ReLU(True),

            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Input : BatchSize * InputChannels * 32 * 32
        # Input : 100 * 3 * 32 * 32
        self.main = nn.Sequential(
# Usage:    nn.ConvTranspose2d(inputChannels, outputChannels, KernelSize, Stride, Padding, bias=False)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # New State : 100 * 64 * 16 * 16
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(),
            
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            # New State : 100 * 128 * 8 * 8
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(),
            
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            # New State : 100 * 256 * 4 * 4
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            # New State : 100 * 512 * 2 * 2
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(),

            nn.Conv2d(ndf*8, 1, 4, 1, 1, bias=False),
            # New State : 100 * 1 * 1 * 1
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)


# Random weights initialization called on for Gen and Disc
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:    # For convolution layers
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1: # For Batch Normalization
        m.weight.data.normal_(1.0, 0.05)
        m.bias.data.fill_(0)

Gen = Generator()
Gen.apply(initialize_weights)

Disc = Discriminator()
Disc.apply(initialize_weights)

input1 = torch.FloatTensor(batchSize, 3, imageSize, imageSize)
input1 = Variable(input1)

noise = torch.FloatTensor(batchSize, nz, 1, 1)
noise = Variable(noise)

# to test on updated weights on Generator -- to generate new Fake image using this noise
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)
fixed_noise = Variable(fixed_noise)

label = torch.FloatTensor(batchSize)
label = Variable(label)

real_label = 1
fake_label = 0

### Binary Cross Entropy Loss ###
criterion = nn.BCELoss()

# setup optionsimizer
Disc_Optimizer = optim.Adam(Disc.parameters(), lr=learning_rate)
Gen_Optimizer = optim.Adam(Gen.parameters(), lr=learning_rate)

for epoch in range(iterations):
    for i, data in enumerate(dataloader, 0):
        
        ###### (A)  Discriminator Network
        ######      Has 2 errors - Real Error and Fake Error

        ####   (A)(i) Train with real data    ####
        ##            Getting real data         ##
        ipt, _ = data
        batch_size = ipt.size(0) ## New batch size for last set which doesn't have 100 elements
        input1.data.resize_(ipt.size()).copy_(ipt)
        ##      Assigning real_label for real 'data'
        label.data.resize_(batch_size).fill_(real_label)

        Disc.zero_grad()
        output = Disc(input1)
        Disc_Real_Err = criterion(output, label)
        Disc_Real_Err.backward()
        real_output = output.data.mean()

        ####   (A)(ii) Train with fake data    ####
        ##             Getting fake data         ##
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        ##      Assigning fake_label for 'noise'
        label.data.fill_(fake_label)

        fake = Gen(noise)
        output = Disc(fake.detach())
        Disc_Fake_Err = criterion(output, label)
        Disc_Fake_Err.backward()
        fake_output = output.data.mean()

        ####    (A)(iii) Maximize log(D(x)) + log(1 - D(G(z)))  ####
        Disc_Err = Disc_Real_Err + Disc_Fake_Err
        Disc_Optimizer.step()
        

        ###### (B)  Generator Network
        ######      Has only 1 errors - Fake Error

        ####   (B)(i) Train with fake data    ####
        Gen.zero_grad()
        label.data.fill_(real_label)  # labels are real for fake data
        output = Disc(fake) ## Goal : make Discriminator give real for fake data
        Gen_Err = criterion(output, label)
        Gen_Err.backward()
        fake_output1 = output.data.mean()
        Gen_Optimizer.step()

        if i%5 == 0:
            print('[%d/%d][%d/%d] D_Loss: %f G_Loss: %f D(real): %f D(G(noise)): %f / %f' 
             	% (epoch, iterations, i, len(dataloader),
                Disc_Err.data[0], Gen_Err.data[0], real_output, fake_output, fake_output1))
        if i % 25 == 0:
            utils.save_image(ipt, '%s/real_epoch_%03d_iter_%03d.png' % (options.out_folder, epoch, i), normalize=True)
            fake = Gen(fixed_noise)
            utils.save_image(fake.data, '%s/fake_epoch_%03d_iter_%03d.png' % (options.out_folder, epoch, i), normalize=True)

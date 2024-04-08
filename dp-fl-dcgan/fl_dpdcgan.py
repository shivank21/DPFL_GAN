from __future__ import print_function

import argparse
import os
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from opacus import PrivacyEngine
from tqdm import tqdm


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-root",default="mnist",  help="path to dataset")
parser.add_argument(
    "--workers", type=int, help="number of data loading workers", default=2
)
parser.add_argument("--batch-size", type=int, default=64, help="input batch size")
parser.add_argument(
    "--imageSize",
    type=int,
    default=64,
    help="the height / width of the input image to network",
)
parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--ngf", type=int, default=128)
parser.add_argument("--ndf", type=int, default=128)
parser.add_argument(
    "--epochs", type=int, default=25, help="number of epochs to train for"
)
parser.add_argument(
    "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
)
parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
)
parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument("--netG", default="", help="path to netG (to continue training)")
parser.add_argument("--netD", default="", help="path to netD (to continue training)")
parser.add_argument(
    "--outf", default=".", help="folder to output images and model checkpoints"
)
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument(
    "--target-digit",
    type=int,
    default=8,
    help="the target digit(0~9) for MNIST training",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="GPU ID for this process (default: 'cuda')",
)
parser.add_argument(
    "--disable-dp",
    action="store_true",
    default=False,
    help="Disable privacy training and just train with vanilla SGD",
)
parser.add_argument(
    "--secure-rng",
    action="store_true",
    default=False,
    help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
)
parser.add_argument(
    "-r",
    "--n-runs",
    type=int,
    default=1,
    metavar="R",
    help="number of runs to average on (default: 1)",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=1.0,
    metavar="S",
    help="Noise multiplier (default 1.0)",
)
parser.add_argument(
    "-c",
    "--max-per-sample-grad_norm",
    type=float,
    default=1.0,
    metavar="C",
    help="Clip per-sample gradients to this norm (default 1.0)",
)
parser.add_argument(
    "--delta",
    type=float,
    default=1e-5,
    metavar="D",
    help="Target delta (default: 1e-5)",
)

opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True


try:
    dataset = dset.MNIST(
        root=opt.data_root,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    idx = dataset.targets == opt.target_digit
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]
    nc = 1
except ValueError:
    print("Cannot load dataset")

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=int(opt.workers),
    batch_size=opt.batch_size,
)

device = torch.device(opt.device)
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.GroupNorm(min(32, ndf * 8), ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 4), ndf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 2), ndf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf), ndf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu)
netG = netG.to(device)
netG.apply(weights_init)
if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG))

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def add_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict["_module."+k] = v
    return new_state_dict

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 2), ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 4), ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 8), ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu)
netD = netD.to(device)
netD.apply(weights_init)
if opt.netD != "":
    netD.load_state_dict(torch.load(opt.netD))
def train_client(dataloader, netG, netD, optimizerG, optimizerD, device, criterion):    
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        label_true = torch.full((batch_size,), REAL_LABEL, device=device)
        
        output = netD(real_data)
        errD_real = criterion(output, label_true)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label_fake = torch.full((batch_size,), FAKE_LABEL, device=device)
        output = netD(fake.detach())
        errD_fake = criterion(output, label_fake)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        optimizerD.step()
        netG.zero_grad()
        
        label_g = torch.full((batch_size,), REAL_LABEL, device=device)
        output_g = netD(fake)
        errG = criterion(output_g, label_g)
        errG.backward()
        D_G_z2 = output_g.mean().item()
        
        optimizerG.step()
n_clients = 2
split_lengths = [len(dataset) // n_clients] * (n_clients - 1)
split_lengths.append(len(dataset) - sum(split_lengths))
client_datasets = torch.utils.data.random_split(dataset, split_lengths)
client_dataloaders = [torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size) for dataset in client_datasets]
criterion = nn.BCELoss()

FIXED_NOISE = torch.randn(opt.batch_size, nz, 1, 1, device=device)
REAL_LABEL = 1.0
FAKE_LABEL = 0.0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if not opt.disable_dp:
    privacy_engine = PrivacyEngine(secure_mode=opt.secure_rng)

    netD, optimizerD, dataloader = privacy_engine.make_private(
        module=netD,
        optimizer=optimizerD,
        data_loader=dataloader,
        noise_multiplier=opt.sigma,
        max_grad_norm=opt.max_per_sample_grad_norm,
    )

optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.epochs):
    state_dictsG = []
    state_dictsD = []

    # Simulate training on each client
    for client_dataloader in client_dataloaders:
        # Creating new instances or copying your models for each client might be necessary
        client_netG = Generator(ngpu).to(device)
        client_netG.load_state_dict(netG.state_dict())
        renamed_state_dict = {}
        for key, value in netD.state_dict().items():
            new_key = key.replace("_module.main", "main")
            renamed_state_dict[new_key] = value

        # Now, use the renamed_state_dict to load the state dict into your model
        client_netD = Discriminator(ngpu).to(device)
        client_netD.load_state_dict(renamed_state_dict)
       
        #client_netD.load_state_dict(netD.state_dict())
        
        client_optimizerG = optim.Adam(client_netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        client_optimizerD = optim.Adam(client_netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        
        train_client(client_dataloader, client_netG, client_netD, client_optimizerG, client_optimizerD, device, criterion)
        
        # Collect model updates
        state_dictsG.append(client_netG.state_dict())
        state_dictsD.append(client_netD.state_dict())
    n_dict={}
    for key, value in netD.state_dict().items():
          new_key = key.replace("_module.main", "main")
          n_dict[new_key] = value  
    # Aggregate and update global models
    with torch.no_grad():
        for key in netG.state_dict():
            updated_weights = torch.mean(torch.stack([sd[key] for sd in state_dictsG]), 0)
            netG.state_dict()[key].copy_(updated_weights)
        print('Example state_dict keys (client):', list(state_dictsD[0].keys())[:])
        print('Expected model keys (global):', list(n_dict.keys()))
        state_dictsD = [remove_module_prefix(sd) for sd in state_dictsD]

        for key, _ in n_dict.items():
            updated_weights = torch.mean(torch.stack([sd[key] for sd in state_dictsD]), 0)
            n_dict[key] = updated_weights
        netD.load_state_dict(add_module_prefix(n_dict))
    

    print(f"Completed Epoch {epoch + 1}/{opt.epochs}")

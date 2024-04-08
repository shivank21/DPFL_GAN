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
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-root", help="path to dataset")
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
    "--device",
    type=str,
    default="cuda",
    help="GPU ID for this process (default: 'cuda')",
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

dataset = dset.CelebFaces(
    root=opt.data_root,
    split="train",
    transform=transforms.Compose(
        [
            transforms.Resize(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers),
)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True
inception_score_metric = InceptionScore()

device = torch.device(opt.device)
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

netG = Generator(ngpu)
netG = netG.to(device)
netG.apply(weights_init)
if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG))

netD = Discriminator(ngpu)
netD = netD.to(device)
netD.apply(weights_init)
if opt.netD != "":
    netD.load_state_dict(torch.load(opt.netD))

criterion = nn.BCELoss()

FIXED_NOISE = torch.randn(opt.batch_size, nz, 1, 1, device=device)
REAL_LABEL = 1.0
FAKE_LABEL = 0.0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
mean_inception_score_list = []
std_inception_score_list = []
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
inception = InceptionScore()

for epoch in range(opt.epochs):
    data_bar = tqdm(dataloader)
    for i, data in enumerate(data_bar, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        optimizerD.zero_grad(set_to_none=True)

        real_data = data[0].to(device)
        batch_size = real_data.size(0)

        # train with real
        label_true = torch.full((batch_size,), REAL_LABEL, device=device)
        output = netD(real_data)
        errD_real = criterion(output, label_true)
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label_fake = torch.full((batch_size,), FAKE_LABEL, device=device)
        output = netD(fake.detach())
        errD_fake = criterion(output, label_fake)

        # below, you actually have two backward passes happening under the hood
        # which opacus happens to treat as a recursive network
        # and therefore doesn't add extra noise for the fake samples
        # noise for fake samples would be unnecesary to preserve privacy

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        optimizerD.zero_grad(set_to_none=True)

        D_G_z1 = output.mean().item()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()

        label_g = torch.full((batch_size,), REAL_LABEL, device=device)
        output_g = netD(fake)
        errG = criterion(output_g, label_g)
        errG.backward()
        D_G_z2 = output_g.mean().item()
        optimizerG.step()

        if not opt.disable_dp:
            epsilon = privacy_engine.accountant.get_epsilon(delta=opt.delta)
            data_bar.set_description(
                f"epoch: {epoch}, Loss_D: {errD.item()} "
                f"Loss_G: {errG.item()} D(x): {D_x} "
                f"D(G(z)): {D_G_z1}/{D_G_z2}"
                "(ε = %.2f, δ = %.2f)" % (epsilon, opt.delta)
            )
        else:
            data_bar.set_description(
                f"epoch: {epoch}, Loss_D: {errD.item()} "
                f"Loss_G: {errG.item()} D(x): {D_x} "
                f"D(G(z)): {D_G_z1}/{D_G_z2}"
            )

        if i % 100 == 0:
            vutils.save_image(
                real_data, "%s/real_samples.png" % opt.outf, normalize=True
            )
            fake = netG(FIXED_NOISE)
            fake_images = ((fake + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            ##fake_images = fake_images(0, 255, (100, 3, 299, 299), dtype=torch.uint8)
            fake_images = fake_images.cpu()
            fake_images = fake_images.repeat(1, 3, 1, 1)
            inception.update(fake_images)
            mean_inception_score, std_inception_score = inception.compute()
            mean_inception_score_list.append(mean_inception_score)
            std_inception_score_list.append(std_inception_score)
            print(f"Mean Inception Score: {mean_inception_score}, Std Inception Score: {std_inception_score}")
            vutils.save_image(
                fake.detach(),
                "%s/fake_samples_epoch_%03d.png" % (opt.outf, epoch),
                normalize=True,
            )
    if epoch % 25 == 0:  # Save checkpoints after every 25 epochs
        torch.save(netG.state_dict(), "%s/netG_epoch_%d.pth" % (opt.outf, epoch))
        torch.save(netD.state_dict(), "%s/netD_epoch_%d.pth" % (opt.outf, epoch))


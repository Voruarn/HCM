import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from tqdm import tqdm
import sys
from models_MGDiT import MGDiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import collections
from torch.utils.tensorboard import SummaryWriter
from datasets.AB2019BASDataset import AB2019BASDataset
from datasets.CBASDataset import CBASDataset
from download import find_model

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['WORLD_SIZE'] = '1'


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """

    logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    print(args)
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Setup DDP:

    rank = 0
    seed = args.global_seed 
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-AB2019"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/CHKP"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    tb_writer = SummaryWriter(experiment_dir)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = MGDiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        sem_channel=1
    )
    
    if args.ckpt:
        ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        checkpoint_model = find_model(ckpt_path)
        model.load_state_dict(checkpoint_model, strict=False)
        print('load model from:', args.ckpt)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"/home/aimb/fyx/Projects/Model_pth/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:

    dataset = CBASDataset(is_train=True,voc_dir=args.data_path)
    sampler = torch.utils.data.RandomSampler(dataset)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    
    logger.info(f"Training for {args.epochs} epochs...")
    cur_epoch=0
    for epoch in range(cur_epoch, args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        data_loader = tqdm(loader, file=sys.stdout)
        running_loss = 0.0
        log_steps = 0

        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            
            # print('y:', y)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            data_loader.desc = "Epoch {}/{}, loss: {:.4f}, step: {}".format(epoch, args.epochs,
                                                running_loss/log_steps, train_steps)
        
        tags = ["train_loss", "learning_rate" ]

        tb_writer.add_scalar(tags[0], (running_loss/log_steps), epoch)
        tb_writer.add_scalar(tags[1], opt.param_groups[0]["lr"], epoch)

      # Save DiT checkpoint:
        if (epoch+1) % args.save_epoch == 0 :
            save_name="{}/latest_{}_bas868.pth".format(checkpoint_dir, args.model.replace('/','-'))
            torch.save(model.state_dict(), save_name)
            logger.info(f"Saved checkpoint to {save_name}")

            if (epoch+1) % 200==0:
                save_name="{}/{}_{}e_bas868.pth".format(checkpoint_dir, args.model.replace('/','-'), (epoch+1))
                torch.save(model.state_dict(), save_name)
                logger.info(f"Saved checkpoint to {save_name}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, 
        default="./Datasets/CBAS/Train/Images/",
        help="path to Dataset")
    parser.add_argument("--results-dir", type=str, 
                        default="./results/")

    parser.add_argument("--model", type=str, choices=list(MGDiT_models.keys()), default="MGDiT-S/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save_epoch", type=int, default=5)
    parser.add_argument("--ckpt", type=str, 
        default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)

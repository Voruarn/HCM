import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models_MGDiT import MGDiT_models
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from PIL import Image
import os
import collections
from datasets.MaskDataset import MaskDataset

def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""
    images = [np.clip(im.permute(1,2,0).numpy(),0,1) for im in images]

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx])
                plt.axis('off')
                idx += 1
    fig.suptitle(title, fontsize=30)
    
    # Showing the figure
    plt.savefig('gen_samples.png', bbox_inches='tight')
    plt.show()
    

def save_images(images, args, names):
    """Shows the provided images as sub-pictures in a square"""
    images = [np.clip(im.permute(1,2,0).numpy(),0,1) for im in images]
    save_path=args.save_path+args.ckpt.split('/')[-1].split('.')[0]+'cfg{}_step{}/'.format(args.cfg_scale, args.num_sampling_steps)
    os.makedirs(save_path, exist_ok=True)
    for i in tqdm(range(len(images))):
        img=images[i]
        # 将NumPy数组转换为PIL图像
        # print('1 img.shape:',img.shape)
        img = Image.fromarray(np.uint8(img*255))
        img.save(save_path+'{}'.format(names[i]))
    print('save images done!!!')

    

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Load model:
    latent_size = args.image_size // 8
    model = MGDiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        sem_channel=1
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    checkpoint_model = find_model(ckpt_path)
    # checkpoint_model_keys = list(checkpoint_model.keys())
    # for k in checkpoint_model_keys:
    #     if 'y_embedder' in k :
    #         print(f"Removing key {k} from pretrained checkpoint")
    #         del checkpoint_model[k]

    model.load_state_dict(checkpoint_model, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"/home/aimb/fyx/Projects/Model_pth/sd-vae-ft-{args.vae}").to(device)
    
    dataset = MaskDataset(args.data_path)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)
    
    for iter, batch in enumerate(tqdm(data_loader)):
        # if iter>=8:
        #     break

        gts, names=batch['mask'], batch['name']
        # for i, name in enumerate(names):
        #     print('No.{}: {}'.format((i+1), name))
        gts = gts.to(device)
        # break

        # Create sampling noise:
        n = args.batch_size
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        # y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        gts = torch.cat([gts, gts], 0)
        # y_null = torch.tensor([1000] * n, device=device)
        # y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=gts, cfg_scale=args.cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        # Save and display images:
        # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
        save_images(samples.cpu(), args, names)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, 
        default='./Gen_Masks/',
        help="path to Dataset")
    parser.add_argument("--model", type=str, choices=list(MGDiT_models.keys()), default="MGDiT-S/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--ckpt", type=str, 
default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="download datasets")
    parser.add_argument("--n_cpu", type=int, default=4,
                        help="download datasets")
    parser.add_argument("--save_path", type=str, 
                    default='./Gen_Imgs/')
    args = parser.parse_args()
    main(args)

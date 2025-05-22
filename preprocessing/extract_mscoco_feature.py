import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
from torchvision.utils import save_image

def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    print(args)

    if args.split == "train":
        datas = MSCOCODatabase(root='assets/datasets/coco/train2014',
                             annFile='assets/datasets/coco/annotations/captions_train2014.json',
                             size=resolution)
        save_dir = f'assets/datasets/coco{resolution}_features/train'
    elif args.split == "val":
        datas = MSCOCODatabase(root='assets/datasets/coco/val2014',
                             annFile='assets/datasets/coco/annotations/captions_val2014.json',
                             size=resolution)
        save_dir = f'assets/datasets/coco{resolution}_features/val'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    os.makedirs(save_dir, exist_ok=True)  # 使用exist_ok以防目录已存在

    autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            x, captions = data

            if len(x.shape) == 3:
                x = x[None, ...]
            
            # 保存原始图像到相同的save_dir目录
            img_tensor = torch.tensor(x, device='cpu')  # 使用CPU保存图像
            save_image(img_tensor, os.path.join(save_dir, f'{idx}.png'))
            
            # 继续原有处理流程
            x = torch.tensor(x, device=device)
            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}.npy'), moments)

            latent = clip.encode(captions)
            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)


if __name__ == '__main__':
    main()
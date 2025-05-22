<h1 align="center"> Preprocessing Guide
</h1>

#### Dataset download
For ImageNet, we follow the preprocessing code in [REPA](https://github.com/sihyun-yu/REPA), which is based on [edm2](https://github.com/NVlabs/edm2).
The range of input is [-1, 1]. 
We currently support 256x256 generation on ImageNet.

After downloading and unzipping ImageNet, please run the following scripts:
```bash
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
python dataset_tools.py convert --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=[TARGET_PATH]/images --resolution=256x256 --transform=center-crop-dhariwal
```

```bash
# Convert the pixel data to VAE latents
python dataset_tools.py encode --source=[TARGET_PATH]/images \
    --dest=[TARGET_PATH]/vae-sd
```

For MS-COCO, we leverage the preprocessing scripts in [U-ViT](https://github.com/baofff/U-ViT/tree/main/scripts). 
First, we download MS-COCO dataset to `/U-ViT/assets/datasets/coco/`.
Then we put our `extract_mscoco_feature.py` and `extract_empty_feature.py` at `U-ViT/`.
## Acknowledgement

This code is mainly built upon [REPA](https://github.com/sihyun-yu/REPA), [edm2](https://github.com/NVlabs/edm2), and [U-ViT](https://github.com/baofff/U-ViT) repository.
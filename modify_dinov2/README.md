<h1 align="center"> Modification Guide
</h1>


### Modifications for DINOv2
At the first run, DINOv2 models will be automatically downloaded from `torch.hub`. Then you can replace the `__init__.py`, `attention.py`, `block.py`, and `vision_transformer.py` with our scripts, which will enable the model to return intermediate attention logits.

If you want to preserve the original `torch.hub` environment for other projects, you can use `export TORCH_HOME` to specify another directory for downloading DINOv2 in advance.
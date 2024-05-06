# QuickTuneTool (WIP)

This repo is based on the QuickTune paper
"**Quick-Tune: Quickly Learning Which Pre Trained Model to Fine Tune and How**" [ICLR2024](https://openreview.net/forum?id=tqh1zdXIra)


## Use QuickTuneTool

### Prepare Environment
To install the QuickTuneTool, you can simply use `pip`:
```bash
pip install quicktunetool
```


### Run on Custom Dataset
The custom dataset must be in Pytorch's [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format, e.g. download the Imagenette dataset:
```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvzf imagenette2-320.tgz
```

```python
from quicktune import QuickTuner, from_pretrained
from quicktune.finetune.finetune_wrapper import eval_finetune_conf

if __name__ == "__main__":
    data_path = "path/to/dataset"
    time_limit = 3600

    # load our pretrained optimizer, the 'mini'-version should be just fine for small dataset
    optimizer = from_pretrained("*mtlbm/mini", num_configs=512)
    qt = QuickTuner(optimizer, eval_finetune_conf)
    qt.fit(data_path, time_limit)
```


## Citation

You can cite our work as follows:

```bib
@inproceedings{
arango2024quicktune,
title={Quick-Tune: Quickly Learning Which Pretrained Model to Finetune and How},
author={Sebastian Pineda Arango and Fabio Ferreira and Arlind Kadra and Frank Hutter and Josif Grabocka},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=tqh1zdXIra}
}
```
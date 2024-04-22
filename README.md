# QuickTune (WIP)
## Quick-Tune: Quickly Learning Which Pre Trained Model to Fine Tune and How [ICLR2024](https://openreview.net/forum?id=tqh1zdXIra)

This repo contains the code for running experiments with QuickTune

![Architecture](figures/figure.svg)


## Run QuickTune

### Prepare Environment
To install QuickTune, you can simply use `pip`:
```bash
pip install quicktune
```


### Run on Custom Dataset
The custom dataset must be in Pytorch's [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format, e.g. download the Imagenette dataset:
```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvzf imagenette2-320.tgz
```

```python
from quicktune import QuickTuner, get_qt_config, setup_quicktune
from quicktune.finetune.utils.eval_autofinetune import eval_finetune_conf

config = get_qt_config("/work/dlclarge2/rapanti-quicktune/datasets/imagenettest10-320", 1000)
ss, md, optimizer = setup_quicktune(config)
QuickTuner(config, optimizer, ss, md, eval_finetune_conf).run()
```

Not implemented
```python
from quicktune import QT_Config, TASK
tuner = QT_Config("path/to/dataset", task=TASK.MTLBM, budget=42, objective_func=...)
tuner.run()
```
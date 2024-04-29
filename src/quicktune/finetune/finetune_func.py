import os

import pandas as pd
import torch
import yaml

from quicktune.finetune import finetune
from quicktune.finetune.utils.build_parser import build_parser
from quicktune.utils.qt_utils import QTaskStatus, QTunerResult


hp_list = [
    "batch_size",
    "bss_reg",
    "clip_grad_norm",
    "cotuning_reg",
    "cutmix",
    "delta_reg",
    "drop",
    "lr",
    "mixup",
    "mixup_prob",
    "model",
    "opt",
    "pct_to_freeze",
    "smoothing",
    "sp_reg",
    "warmup_epochs",
    "warmup_epochs",
    "warmup_lr",
    "weight_decay",
]

bool_hp_list = [
    "linear_probing",
    "stoch_norm",
]

cond_hp_list = [
    "auto_augment",
    "data_augmentation",
    "layer_decay",
    "sched",
]

static_args = [
    "--pretrained",
    "--checkpoint_hist", "1",
    "--epochs", "50",
    "--workers", "8",
]

task_args = [
    "train-split",
    "val-split",
    "num_classes",
]


def eval_finetune_conf(config: dict):

    data_path = config["data_path"]
    budget = config["budget"]
    task_info = config["task_info"]
    experiment = config["experiment"]
    output = config["output"]
    verbose = config.get("verbose", False)
    hp_config: dict = config["hp_config"]
    hp_config.pop("amp", None)

    args = [data_path]

    for hp in hp_list:
        if hp in hp_config:
            args += [f"--{hp}", str(hp_config[hp])]

    for hp in bool_hp_list:
        enabled = hp_config.get(hp, False)
        if enabled:
            args += [f"--{hp}"]

    for hp in cond_hp_list:
        option = hp_config.get(hp, "None")
        if option != "None":
            args += [f"--{hp}", str(option)]

    opt_betas = hp_config.get("opt_betas", "None")
    if opt_betas != "None":
        opt_betas = opt_betas.strip("[]").split(",")
        args += ["--opt_betas", *opt_betas]

    for arg in task_args:
        args += [f"--{arg}", str(task_info[arg])]

    args += ["--epochs_step", str(budget)]
    args += ["--experiment", experiment]
    args += ["--output", output]

    output_dir = os.path.join(output, experiment)
    resume_path = os.path.join(output_dir, "last.pth.tar")
    if os.path.exists(resume_path):
        args += ["--resume", resume_path]
    
    args += static_args

    parser = build_parser()
    args, _ = parser.parse_known_args(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    args.verbose = True if verbose else False

    try:
        torch.cuda.empty_cache()
        finetune.main(args, args_text)
    except Exception as e:
        if verbose:
            print("Error:", e)
        result = QTunerResult(
            score=-1,
            time=-1,
            status=QTaskStatus.ERROR,
            info=str(e),
        )
        return result

    # read last line of txt
    summary = pd.read_csv(os.path.join(output_dir, "summary.csv"))
    eval_top1 = summary["eval_top1"].iloc[-1]
    eval_time = summary["eval_time"].iloc[-1]

    result = QTunerResult(
        score=eval_top1,
        time=eval_time,
        status=QTaskStatus.SUCCESS,
    )

    return result

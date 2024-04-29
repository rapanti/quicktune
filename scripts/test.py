from quicktune.tuners import QuickTuner
from quicktune.configs import get_qt_config
from quicktune.finetune.finetune_func import eval_finetune_conf

if __name__ == "__main__":
    config = get_qt_config()
    qt = QuickTuner(config, eval_finetune_conf)
    qt.fit("/home/evilknivl/projects/datasets/imagenettest", time_limit=3600)

from quicktune import QuickTuner, get_qt_config, setup_quicktune
from quicktune.finetune.utils.finetune_func import eval_finetune_conf

if __name__ == "__main__":
    config = get_qt_config("/work/dlclarge2/rapanti-quicktune/datasets/imagenettest10-320", 1000)
    ss, md, optimizer = setup_quicktune(config)
    QuickTuner(config, optimizer, ss, md, eval_finetune_conf).run()

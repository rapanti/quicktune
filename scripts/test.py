from quicktune import QuickTuner, get_qt_config, setup_quicktune
from quicktune.finetune.utils.eval_autofinetune import eval_finetune_conf

if __name__ == "__main__":
    config = get_qt_config("/home/evilknivl/projects/datasets/imagenettest", 1000)
    ss, md, optimizer = setup_quicktune(config)
    QuickTuner(config, optimizer, ss, md, eval_finetune_conf).run()

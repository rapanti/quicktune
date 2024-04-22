from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    OrConjunction,
)
from ConfigSpace.read_and_write import json as cs_json

data_augmentation = ["None", "trivial_augment", "random_augment", "auto_augment"]
models = [
    "beit_base_patch16_384",
    "beit_large_patch16_512",
    "convnext_small_384_in22ft1k",
    "deit3_small_patch16_384_in21ft1k",
    "dla46x_c",
    "edgenext_small",
    "edgenext_x_small",
    "edgenext_xx_small",
    "mobilevit_xs",
    "mobilevit_xxs",
    "mobilevitv2_075",
    "swinv2_base_window12to24_192to384_22kft1k",
    "tf_efficientnet_b4_ns",
    "tf_efficientnet_b6_ns",
    "tf_efficientnet_b7_ns",
    "volo_d1_384",
    "volo_d3_448",
    "volo_d4_448",
    "volo_d5_448",
    "volo_d5_512",
    "xcit_nano_12_p8_384_dist",
    "xcit_small_12_p8_384_dist",
    "xcit_tiny_12_p8_384_dist",
    "xcit_tiny_24_p8_384_dist",
]
schd = [
    "None",
    "cosine",
    "multistep",
    "plateau",
    "step",
]

amp = Categorical("amp", items=(False, True), ordered=True)
aa = Categorical("auto_augment", items=("v0", "original"))
bs = Categorical("batch_size", items=(2, 4, 8, 16, 32, 64), ordered=True)
bss = Categorical("bss_reg", items=(0.0, 0.0001, 0.001, 0.01, 0.1), ordered=True)
cg = Categorical("clip_grad", items=("None", 1, 10), ordered=True)
cr = Categorical("cotuning_reg", items=(0.0,), ordered=True)
cm = Categorical("cutmix", items=(0.0, 0.1, 0.25, 0.5, 1, 2, 4), ordered=True)
da = Categorical("data_augmentation", items=data_augmentation)
de = Categorical("decay_epochs", items=(10, 20), ordered=True)
dr = Categorical("decay_rate", items=(0.1, 0.5), ordered=True)
dreg = Categorical("delta_reg", items=(0.0, 0.0001, 0.001, 0.01, 0.1), ordered=True)
drop = Categorical("drop", items=(0.0, 0.1, 0.2, 0.3, 0.4), ordered=True)
ep = Constant("epochs", value=50)
ld = Categorical("layer_decay", items=("None", 0.65, 0.75), ordered=True)
lp = Categorical("linear_probing", items=(False, True), ordered=True)
lr = Categorical("lr", items=(1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01), ordered=True)
mu = Categorical("mixup", items=(0.0, 0.2, 0.4, 1.0, 2.0, 4.0, 8.0), ordered=True)
mu_prob = Categorical("mixup_prob", items=(0.0, 0.25, 0.5, 0.75, 1.0), ordered=True)
m = Categorical("model", items=models)
mom = Categorical("momentum", items=(0.0, 0.8, 0.9, 0.95, 0.99), ordered=True)
opt = Categorical("opt", items=("sgd", "momentum", "adam", "adamw", "adamp"))
ob = Categorical(
    "opt_betas", items=("[0.9, 0.999]", "[0, 0.99]", "[0.9, 0.99]", "[0, 0.999]")
)
pe = Categorical("patience_epochs", items=(2, 5), ordered=True)
ptf = Categorical("pct_to_freeze", items=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), ordered=True)
ram = Categorical("ra_magnitude", items=(9, 17), ordered=True)
rno = Categorical("ra_num_ops", items=(2, 3), ordered=True)
sched = Categorical("sched", items=schd)
sm = Categorical("smoothing", items=(0.0, 0.05, 0.1), ordered=True)
sr = Categorical("sp_reg", items=(0.0, 0.0001, 0.001, 0.01, 0.1), ordered=True)
sn = Categorical("stoch_norm", items=(False, True), ordered=True)
we = Categorical("warmup_epochs", items=(0, 5, 10), ordered=True)
wlr = Categorical("warmup_lr", items=(0.0, 1e-5, 1e-6), ordered=True)
wd = Categorical("weight_decay", items=(0.1, 0.01, 1e-3, 1e-5, 1e-4, 0), ordered=True)

hps = [
    amp,
    aa,
    bs,
    bss,
    cg,
    cr,
    cm,
    da,
    de,
    dr,
    dreg,
    drop,
    ep,
    ld,
    lp,
    lr,
    mu,
    mu_prob,
    m,
    mom,
    opt,
    ob,
    pe,
    ptf,
    ram,
    rno,
    sched,
    sm,
    sr,
    sn,
    we,
    wlr,
    wd,
]
cs = ConfigurationSpace()
cs.add_hyperparameters(hps)

conditions = [
    EqualsCondition(aa, da, "auto_augment"),
    EqualsCondition(mom, opt, "momentum"),
    EqualsCondition(pe, sched, "plateau"),
    EqualsCondition(ram, da, "random_augment"),
    EqualsCondition(rno, da, "random_augment"),
    OrConjunction(
        EqualsCondition(de, sched, "step"),
        EqualsCondition(de, sched, "multistep"),
    ),
    OrConjunction(
        EqualsCondition(dr, sched, "step"),
        EqualsCondition(dr, sched, "multistep"),
    ),
    OrConjunction(
        EqualsCondition(ob, opt, "adam"),
        EqualsCondition(ob, opt, "adamw"),
        EqualsCondition(ob, opt, "adamp"),
    ),
]
cs.add_conditions(conditions)

cs_string = cs_json.write(cs)
with open("mtlbm.json", "w") as f:
    f.write(cs_string)

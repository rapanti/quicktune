import yaml

with open(
    "/home/evilknivl/projects/quicktunetool/src/quicktune/configs/cv/classification/config.yaml",
    "r",
) as f:
    config = yaml.safe_load(f)

    print(config)
# config = {
#     "meta-dataset": {
#         "root": "/home/evilknivl/projects/edit-qt-mds/mtlbm-metadataset",
#         "version": "mini",
#         "standardize_num_args": True,
#         "model_args_first": True,
#         "load_only_dataset_descriptors": True,
#     },
#     "surrogate_config": {
#         "feature_extractor": {
#             "hidden_dim": 128,
#             "output_dim": 128,
#             "in_curves_dim": 1,
#             "out_curves_dim": 128,
#             "in_metafeatures_dim": 4,
#             "out_metafeatures_dim": 16,
#             "encoder_num_layers": 2,
#             "encoder_dim_ranges": (24, 69),
#         },
#         "cost_predictor": {
#             "hidden_dim": 128,
#             "output_dim": 1,
#             "in_curves_dim": 1,
#             "out_curves_dim": 128,
#             "in_metafeatures_dim": 4,
#             "out_metafeatures_dim": 16,
#             "encoder_num_layers": 2,
#             "encoder_dim_ranges": (24, 69),
#         },
#         "include_metafeatures": True,
#         "cost_aware": True,
#         "lr": 1e-3,
#     },
# }

# with open("config.yaml", "w") as f:
#     yaml.dump(config, f)
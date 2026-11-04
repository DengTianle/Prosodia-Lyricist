"""Configuration loading without global state or working-directory assumptions."""

from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section, keys in {
        "data": ("dali_dir", "lakh_file", "prepared_dir", "song_ids_file", "pretraining_manifest",
                 "base_prepared_dir"),
        "model": ("melody_checkpoint", "template_checkpoint", "xai_tokenizer_dir",
                  "xai_bart_config", "xai_cache_dir"),
        "training": ("output_dir",),
    }.items():
        for key in keys:
            if not config.get(section, {}).get(key):
                continue
            value = Path(config[section][key]).expanduser()
            config[section][key] = str((path.parent / value).resolve())
    pretrained = config["model"].get("pretrained", "")
    fusion = config["model"].get("contrastive_fusion")
    if isinstance(fusion, dict) and fusion.get("checkpoint"):
        config["model"]["contrastive_fusion"]["checkpoint"] = str(
            (path.parent / Path(fusion["checkpoint"]).expanduser()).resolve()
        )
    if pretrained.startswith((".", "/", "~")):
        config["model"]["pretrained"] = str((path.parent / Path(pretrained).expanduser()).resolve())
    return config

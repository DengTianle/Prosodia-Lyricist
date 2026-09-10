"""Configuration loading without global state or working-directory assumptions."""

from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section, keys in {
        "data": ("dali_dir", "prepared_dir", "song_ids_file"),
        "training": ("output_dir",),
    }.items():
        for key in keys:
            if not config[section].get(key):
                continue
            value = Path(config[section][key]).expanduser()
            config[section][key] = str((path.parent / value).resolve())
    pretrained = config["model"]["pretrained"]
    if pretrained.startswith((".", "/", "~")):
        config["model"]["pretrained"] = str((path.parent / Path(pretrained).expanduser()).resolve())
    return config

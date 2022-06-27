import yaml
from box import Box

def load_yml(path="config.yml"):
    with open(path, "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
    return cfg

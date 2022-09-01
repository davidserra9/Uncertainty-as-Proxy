# -*- coding: utf-8 -*-
"""
This module contains the functions for managing external files as the configuration files
@author: David Serrano Lozano, @davidserra9
"""
import pickle
import torch
import yaml
from box import Box

def load_yml(path="config.yml"):
    with open(path, "r") as ymlfile:
        file = yaml.safe_load(ymlfile)
        file['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = Box(file)
    return cfg

def load_pickle(path):
    with open(path, "rb") as f:
        p = pickle.load(f)
    return p

def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

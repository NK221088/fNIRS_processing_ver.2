import numpy as np

def compute_effect_size(class_instance):
    raw_epochs = class_instance.all_raw_epochs
    preprocessed_epochs = class_instance.all_epochs
    
    effect_size_raw = 0
    effect_size_preprocessed = 0
    return effect_size_raw, effect_size_preprocessed
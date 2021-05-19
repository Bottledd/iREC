import torch


def convert_flattened_indices(indices, beamwidth):
    selected_beams = indices % beamwidth
    selected_samples = indices // beamwidth

    return selected_beams, selected_samples
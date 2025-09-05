import numpy as np
import os
import json

def load_sample_data(sample_path):
    rx = np.load(os.path.join(sample_path, 'rx.npy'))
    with open(os.path.join(sample_path, 'meta.json'), 'r') as f:
        meta = json.load(f)
    return rx, meta

def save_decoded_bits(bits, sample_path):
    output_path = os.path.join(sample_path, 'decoded_bits.npy')
    np.save(output_path, bits.astype(int))
    return output_path

def calculate_ber(decoded_bits, true_bits):
    decoded_bits = np.array(decoded_bits)
    true_bits = np.array(true_bits)
    min_len = min(len(decoded_bits), len(true_bits))
    if min_len == 0:
        return None
    errors = np.sum(decoded_bits[:min_len] != true_bits[:min_len])
    ber = errors / min_len
    return ber

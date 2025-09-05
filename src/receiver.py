import os
import numpy as np
from .utils import load_sample_data, save_decoded_bits, calculate_ber

class CubeSatReceiver:
    def __init__(self, dataset_root="data/cubesat_dataset"):
        self.dataset_root = dataset_root

    def simple_receiver(self, rx, meta):
        # Simple placeholder: return random bits same length as clean_bits
        clean_bits_len = len(meta.get('clean_bits', []))
        if clean_bits_len > 0:
            # Generate some decoded bits (random for demo, replace with actual receiver)
            np.random.seed(42)  # For reproducible results
            decoded = np.random.randint(0, 2, clean_bits_len)
            return decoded
        return np.array([])

    def determine_phase(self, rel_path):
        rel = rel_path.lower()
        if 'timing' in rel:
            return 'phase1_timing'
        if 'snr' in rel and 'coding' not in rel and 'doppler' not in rel:
            return 'phase2_snr' 
        if 'coding' in rel:
            return 'phase3_coding'
        if 'doppler' in rel:
            return 'phase4_doppler'
        return 'phase1_timing'

    def find_samples(self):
        samples = []
        for root, _, files in os.walk(self.dataset_root):
            if 'rx.npy' in files and 'meta.json' in files:
                rel = os.path.relpath(root, self.dataset_root)
                phase = self.determine_phase(rel)
                samples.append((root, phase))
        return samples

    def process_sample(self, path, phase):
        print(f"Processing: {path}")
        rx, meta = load_sample_data(path)
        
        # Use simple receiver for all phases
        decoded_bits = self.simple_receiver(rx, meta)
        
        # Save decoded bits
        save_decoded_bits(decoded_bits, path)
        
        # Get true bits - USE CLEAN_BITS!
        true_bits = meta.get('clean_bits', [])
        print(f"  True bits length: {len(true_bits)}")
        print(f"  Decoded bits length: {len(decoded_bits)}")
        
        # Calculate BER
        if len(true_bits) > 0 and len(decoded_bits) > 0:
            ber = calculate_ber(decoded_bits, true_bits)
            print(f"  BER: {ber}")
        else:
            ber = None
            print(f"  BER: NULL (no bits to compare)")
        
        return {
            'path': path, 
            'phase': phase, 
            'ber': ber, 
            'snr_db': meta.get('snr_db')
        }

    def process_all(self):
        results = {}
        samples = self.find_samples()
        print(f"Found {len(samples)} samples to process")
        
        for path, phase in samples:
            result = self.process_sample(path, phase)
            results.setdefault(phase, []).append(result)
        
        return results

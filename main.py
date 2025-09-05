import os
import json
from src.receiver import CubeSatReceiver

def main():
    dataset_root = 'data/cubesat_dataset'
    if not os.path.exists(dataset_root):
        print(f"Dataset not found at {dataset_root}")
        return

    receiver = CubeSatReceiver(dataset_root)
    results = receiver.process_all()

    os.makedirs('results', exist_ok=True)
    with open('results/processing_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n=== RESULTS SUMMARY ===")
    for phase, entries in results.items():
        print(f"{phase}:")
        for entry in entries:
            print(f"  BER: {entry['ber']}, SNR: {entry['snr_db']} dB")
    
    print(f"\nResults saved to 'results/processing_results.json'")

if __name__ == '__main__':
    main()

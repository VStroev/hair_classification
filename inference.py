import argparse
import csv

import dlib
import torch

from data import ValDataset
from model import HairDetectorModel


def apply_model(model, img, threshold):
    if img is None:
        return -1
    result_probs = model(img.unsqueeze(0))
    result_probs = result_probs > threshold
    return int(result_probs[0][0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=False, default=.5)
    args = parser.parse_args()

    detector = dlib.get_frontal_face_detector()
    val_dataset = ValDataset(args.data_dir, detector)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = HairDetectorModel().to(device)

    state = torch.load(args.model_path)
    model.load_state_dict(state)

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
    )

    with open('result.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(val_dataset)):
            img, path = val_dataset[i]
            res = apply_model(quantized_model, img, args.threshold)
            writer.writerow([path, res])

if __name__ == '__main__':
    main()

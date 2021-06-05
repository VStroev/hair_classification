import argparse
from os.path import join
import dlib
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data import HairDataset, ValDataset
from model import HairDetectorModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    detector = dlib.get_frontal_face_detector()
    train_datast = HairDataset(join(args.data_dir, "data256x256_longhair"),
                               join(args.data_dir, "data256x256_shorthair"),
                               detector)
    val_dataset = ValDataset(join(args.data_dir, "hair-val"), detector)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = HairDetectorModel().to(device)

    trainer = pl.Trainer()
    trainer.fit(model, DataLoader(train_datast, batch_size=32, shuffle=True), DataLoader(val_dataset))

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), 'result_model.pth')

if __name__ == '__main__':
    main()
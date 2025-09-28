#!/usr/bin/env python3
import argparse, glob, torch
from sketch import SketchDataset, SketchRNN, train_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True, help='Paths to .npy files (QuickDraw format).')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=256)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = SketchDataset(args.data, data_type='train', Nmax=100)
    model = SketchRNN().to(device)
    train_model(model, ds, device=device, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()

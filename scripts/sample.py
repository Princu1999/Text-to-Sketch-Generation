#!/usr/bin/env python3
import argparse, torch, numpy as np
from sketch import SketchRNN, sample_sketch, plot_sketch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=False, help='Path to model checkpoint')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SketchRNN().to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    strokes = sample_sketch(model, device=device)
    plot_sketch(strokes)

if __name__ == "__main__":
    main()

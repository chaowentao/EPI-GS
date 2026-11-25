import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import WRAP_FORWARDS, get_encoder_net, get_noLE_combo_decoder, validate

ROOT = Path(__file__).resolve().parent
PREFIXES = {
    "DistgSSR": ("DistgSSR", "Distg"),
    "EPIT": ("EPIT", "EPIT"),
    "InterNet": ("InterNet", "InterNet"),
    "IINet": ("IINet", "IINet"),
    "LF_DET": ("LF-DET", "LF_DET"),
    "LFT": ("LFT", "LFT"),
}

def load_model(encoder, variant, weights, device, checkpoint_path=None):
    folder, prefix = PREFIXES[encoder]
    path = checkpoint_path or weights / folder / f"{prefix}_{variant}SR_5x5.pth.tar"
    model = get_encoder_net(encoder, 5)
    decoder = get_noLE_combo_decoder(5, 64, 128, 2, 3.0, 1.0)
    if hasattr(model, "upsampling"):
        model.upsampling = decoder
    else:
        model.upsample = decoder
    WRAP_FORWARDS[encoder](model)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(
        {k.removeprefix("module."): v for k, v in checkpoint.get("state_dict", checkpoint).items()},
        strict=True,
    )
    return model.to(device).eval(), path

def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained EPI-GS.")
    parser.add_argument("--encoder", choices=PREFIXES, default="DistgSSR")
    parser.add_argument("--variant", choices=["2x", "4x", "Arbitrary"], default="Arbitrary")
    parser.add_argument("--weights", type=Path, default=ROOT / "saved_models")
    parser.add_argument("--checkpoint", type=Path, help="Override the selected checkpoint")
    parser.add_argument("--test_root", type=Path, default=ROOT / "datasets/test")
    parser.add_argument("--scales", nargs="+", type=int, choices=[2, 3, 4, 6, 8])
    parser.add_argument("--patchsize", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    scales = args.scales or ([2, 3, 4, 6, 8] if args.variant == "Arbitrary" else [int(args.variant[0])])
    if args.variant != "Arbitrary" and scales != [int(args.variant[0])]:
        parser.error("Fixed-scale checkpoints must be tested at their designated scale.")
    torch.set_num_threads(4)
    device = torch.device("cpu" if args.check_only else args.device)
    net, path = load_model(args.encoder, args.variant, args.weights, device, args.checkpoint)
    print(f"Strict checkpoint load: {path}", flush=True)
    if args.check_only:
        return
    report = {"encoder": args.encoder, "variant": args.variant, "checkpoint": path.name,
              "patchsize": args.patchsize, "stride": args.stride, "scales": {}}
    for scale in scales:
        directory = args.test_root / f"SR_5x5_{scale}x"
        p, s, names = validate(net, 5, args.patchsize, args.stride, str(directory) + "/", device, scale)
        values = {"PSNR": float(np.mean(p)), "SSIM": float(np.mean(s)),
                  "datasets": {n: {"PSNR": float(a), "SSIM": float(b)} for n, a, b in zip(names, p, s)}}
        report["scales"][scale] = values
        print(f"x{scale}: {values['PSNR']:.4f} dB / {values['SSIM']:.6f}", flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_suffix(args.report.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.report)

if __name__ == "__main__":
    main()

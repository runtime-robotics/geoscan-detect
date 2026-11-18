"""Export trained weights for Pioneer Code or the standalone RKNN detector."""
import argparse
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('--format', choices=['onnx', 'rknn'], default='onnx')
    parser.add_argument('--chip', help='Exact Rockchip target, read from the drone device tree')
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error('Weights file does not exist')
    if args.format == 'rknn' and (not args.chip or platform.machine() != 'x86_64'
                                   or platform.system() != 'Linux'):
        parser.error('RKNN export requires x86_64 Linux and --chip, e.g. rk3588')
    from ultralytics import YOLO
    options = dict(format=args.format, imgsz=args.imgsz, batch=1, device='cpu')
    if args.format == 'onnx':
        options.update(opset=19, dynamic=False, simplify=True, nms=False)
    else:
        options.update(name=args.chip)
    print(YOLO(str(args.weights)).export(**options))


if __name__ == '__main__':
    main()

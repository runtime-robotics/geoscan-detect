"""Train the single-class detector on a YOLO detection dataset."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='config/rover.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cpu', help='cpu or CUDA index, e.g. 0')
    args = parser.parse_args()
    from ultralytics import YOLO
    import yaml

    data_path = Path(args.data).resolve()
    data = yaml.safe_load(data_path.read_text())
    # Ultralytics otherwise resolves relative roots against its global datasets_dir.
    data['path'] = str((data_path.parent / data['path']).resolve())
    resolved = Path('runs/rover-data.yaml')
    resolved.parent.mkdir(exist_ok=True)
    resolved.write_text(yaml.safe_dump(data))
    model = YOLO('yolo11n.pt')
    model.train(data=str(resolved.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project='runs', name='rover', seed=42)
    print('Weights:', Path(model.trainer.best).resolve())


if __name__ == '__main__':
    main()

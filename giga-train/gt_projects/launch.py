import argparse
import os
import sys
from pathlib import Path


def init_paths(project_name: str) -> None:
    """Make sibling packages importable when running from a source checkout."""
    train_root = Path(__file__).resolve().parents[1]
    workspace_root = train_root.parent
    python_paths = (
        Path(__file__).resolve().parent / project_name,
        train_root,
        workspace_root / 'giga-models',
        workspace_root / 'giga-datasets',
    )
    for python_path in reversed(python_paths):
        path = str(python_path)
        if path not in sys.path:
            sys.path.insert(0, path)

    # DeepSpeed/Accelerate starts fresh Python interpreters. Unlike sys.path,
    # PYTHONPATH is inherited by those worker processes.
    inherited_paths = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    inherited_paths = [path for path in inherited_paths if path]
    source_paths = [str(path) for path in python_paths]
    os.environ['PYTHONPATH'] = os.pathsep.join(
        dict.fromkeys(source_paths + inherited_paths)
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Launch DreamTransfer training.')
    parser.add_argument('--project', default='dream_transfer')
    parser.add_argument(
        '--config',
        default='configs.dream_transfer_full_multiview.config',
    )
    parser.add_argument('--runner', default='dream_transfer.DreamTransferFullTrainer')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_paths(args.project)

    from giga_train import launch_from_config

    launch_from_config(args.config, args.runner)


if __name__ == '__main__':
    main()

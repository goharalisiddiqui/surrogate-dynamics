#!/usr/bin/env python3
"""Command-line interface for the surrogate_dynamics package."""

import argparse
import sys


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=str,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )


def _run_train(args: argparse.Namespace) -> None:
    from surrogate_dynamics.trainer import main as train_main

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["surrogate-dynamics-train", "--config", args.config]
        if args.debug:
            sys.argv.append("--debug")
        train_main()
    finally:
        sys.argv = original_argv

def _run_prepare_dmod(args: argparse.Namespace) -> None:
    """
    Execute the prepare datamodule command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    from surrogate_dynamics.prepare_dmod import main as prepare_dmod_main
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['surrogate-dynamics-prepare_dmod', '--config', args.config]
        if args.debug:
            sys.argv.append('--debug')
        prepare_dmod_main()
    finally:
        sys.argv = original_argv

def _run_predict(args: argparse.Namespace) -> None:
    from surrogate_dynamics.predictor import main as predict_main

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["surrogate-dynamics-predict", "--config", args.config]
        if args.debug:
            sys.argv.append("--debug")
        predict_main()
    finally:
        sys.argv = original_argv


def main_prepare_dmod() -> None:
    """
    Entry point for the surrogate-dynamics-prepare_dmod command.
    """
    parser = argparse.ArgumentParser(
        description="Prepare data module for surrogate dynamics model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_common_args(parser)
    _run_prepare_dmod(parser.parse_args())


def main_train() -> None:
    parser = argparse.ArgumentParser(
        description="Train a surrogate dynamics model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(parser)
    _run_train(parser.parse_args())


def main_predict() -> None:
    parser = argparse.ArgumentParser(
        description="Run predictions with a surrogate dynamics model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(parser)
    _run_predict(parser.parse_args())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Surrogate Dynamics CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a model")
    _add_common_args(train_parser)
    train_parser.set_defaults(func=_run_train)
    
    prepare_parser = subparsers.add_parser('prepare', help='Prepare data module')
    _add_common_args(prepare_parser)
    prepare_parser.set_defaults(func=_run_prepare_dmod)

    predict_parser = subparsers.add_parser("predict", help="Run predictions")
    _add_common_args(predict_parser)
    predict_parser.set_defaults(func=_run_predict)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

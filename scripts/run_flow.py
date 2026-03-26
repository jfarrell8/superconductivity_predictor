"""
scripts/run_flow.py
────────────────────
CLI entry point that triggers the Prefect training_pipeline flow.

Supports two modes:
  • Local (default): runs the flow in-process with no server — identical
    behaviour to run_pipeline.py but with Prefect's task retry / logging.
  • Deployed: submits a run to a live Prefect server/Cloud work pool.

Usage
─────
    # Local, with config defaults (50 trials)
    python scripts/run_flow.py

    # Local, quick smoke test (5 trials)
    python scripts/run_flow.py --trials 5

    # Submit to a deployed work pool (requires `prefect deploy` first)
    python scripts/run_flow.py --deploy --name weekly-retraining

    # Custom config
    python scripts/run_flow.py --config configs/experiment_v1.yaml --trials 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def start_prefect_server() -> subprocess.Popen:
    """Start a local Prefect server in a background subprocess."""
    import subprocess
    import time
    import urllib.request

    logger.info("Starting local Prefect server...")
    server_process = subprocess.Popen(
        ["prefect", "server", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait until the server is actually ready to accept connections
    max_wait = 30  # seconds
    for i in range(max_wait):
        try:
            urllib.request.urlopen("http://localhost:4200/api/health")
            logger.info(f"Prefect server ready at http://localhost:4200")
            return server_process
        except Exception:
            time.sleep(1)
            if i % 5 == 0 and i > 0:
                logger.info(f"Still waiting for Prefect server... ({i}s)")

    raise RuntimeError("Prefect server did not start within 30 seconds.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Prefect training pipeline flow."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_v1.yaml",
        help="Path to YAML experiment config.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override n_trials from config for quick runs.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Submit to a deployed Prefect work pool instead of running locally.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="weekly-retraining",
        help="Deployment name to trigger (only used with --deploy).",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Skip starting a local Prefect server (run in-process instead).",
    )
    args = parser.parse_args()

    server_process = None

    if not args.deploy and not args.no_server:
        import os
        os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
        server_process = start_prefect_server()

    try:
        if args.deploy:
            from prefect.deployments import run_deployment
            print(f"Submitting run to deployment: superconductivity-training-pipeline/{args.name}")
            run = run_deployment(
                name=f"superconductivity-training-pipeline/{args.name}",
                parameters={
                    "config_path": args.config,
                    "n_trials_override": args.trials,
                },
            )
            print(f"Run submitted: {run.id}")
        else:
            from src.orchestration.flows import training_pipeline
            result = training_pipeline(
                config_path=args.config,
                n_trials_override=args.trials,
            )
            print("\n── Pipeline complete ──────────────────────────────────────")
            for k, v in result.items():
                print(f"  {k}: {v}")
    finally:
        # Clean up the server process when the flow finishes
        if server_process is not None:
            logger.info("Shutting down local Prefect server...")
            server_process.terminate()
            server_process.wait()

# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Run the Prefect training pipeline flow."
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="configs/experiment_v1.yaml",
#         help="Path to YAML experiment config.",
#     )
#     parser.add_argument(
#         "--trials",
#         type=int,
#         default=None,
#         help="Override n_trials from config for quick runs.",
#     )
#     parser.add_argument(
#         "--deploy",
#         action="store_true",
#         help="Submit to a deployed Prefect work pool instead of running locally.",
#     )
#     parser.add_argument(
#         "--name",
#         type=str,
#         default="weekly-retraining",
#         help="Deployment name to trigger (only used with --deploy).",
#     )
#     args = parser.parse_args()

#     if args.deploy:
#         # ── Remote: submit a run to a live Prefect deployment ───────────────
#         from prefect.deployments import run_deployment
#         print(f"Submitting run to deployment: superconductivity-training-pipeline/{args.name}")
#         run = run_deployment(
#             name=f"superconductivity-training-pipeline/{args.name}",
#             parameters={
#                 "config_path": args.config,
#                 "n_trials_override": args.trials,
#             },
#         )
#         print(f"Run submitted: {run.id}")
#     else:
#         # ── Local: run in-process ────────────────────────────────────────────
#         from src.orchestration.flows import training_pipeline
#         result = training_pipeline(
#             config_path=args.config,
#             n_trials_override=args.trials,
#         )
#         print("\n── Pipeline complete ──────────────────────────────────────")
#         for k, v in result.items():
#             print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

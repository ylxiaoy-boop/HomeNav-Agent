"""HomeNav-Agent application entry point."""

from __future__ import annotations

from config.settings import config
from interfaces.cli import run_cli
from utils.logger import configure_logging


def main() -> int:
    configure_logging(config.get("system.log_level", "INFO"))
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())

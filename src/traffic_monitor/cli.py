"""Command-line interface for Traffic Monitor."""

import click
from loguru import logger
from .main_supervisor import main as supervisor_main


@click.command()
@click.option(
    "--config",
    "-c",
    help="Path to configuration file",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(config, verbose):
    """Start the Traffic Monitor system."""
    if verbose:
        logger.info("Verbose mode enabled")
    
    if config:
        logger.info(f"Using config file: {config}")
        # TODO: Pass config to supervisor when we refactor config loading
    
    supervisor_main()


if __name__ == "__main__":
    main() 
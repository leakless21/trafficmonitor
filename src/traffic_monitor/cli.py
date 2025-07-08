"""Command-line interface for Traffic Monitor."""

import click
import sys
import yaml
from loguru import logger
from .main_supervisor import main as supervisor_main
from pathlib import Path  # Needed for output directory handling
import time  # For timestamped output folder


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML is not installed. Please install it to use YAML config files.")
            sys.exit(1)
        return yaml.safe_load(f)


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
    help="Enable verbose logging (DEBUG level)",
)
@click.option(
    "--log-level",
    help="Set specific log level (DEBUG, INFO, WARNING, ERROR)",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress non-error output (WARNING level)",
)
@click.option(
    "--mode",
    type=click.Choice(["live", "offline", "eval"]),
    default=None,
    show_default=False,
    help="Run in live (camera/RTSP), offline (full video files), or eval (short window) mode",
)
@click.option(
    "--source",
    "-s",
    default=None,
    help="live: camera index/RTSP | offline/eval: video file or directory",
)
@click.option(
    "--count-line",
    "-l",
    multiple=True,
    help="Counting line coordinates as x1,y1,x2,y2. Repeat for multiple lines or pass 'none' to disable counting",
)
@click.option(
    "--start-sec",
    type=float,
    default=None,
    help="Start processing at this second in the video (evaluation mode)",
)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Process at most this many frames (evaluation mode)",
)
def main(config, verbose, log_level, quiet, mode, source, count_line, start_sec, max_frames):
    """Start the Traffic Monitor system."""
    # Handle log levels
    if quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    elif verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level=log_level)
    
    # -------------------------------------------------------------
    # Load base configuration (if any YAML provided)
    # -------------------------------------------------------------
    config_data = None
    if config:
        logger.info(f"Using config file: {config}")
        try:
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            sys.exit(1)

    # Ensure we have a dict to work with
    if config_data is None:
        config_data = {}

    # -------------------------------------------------------------
    # Interactive prompts for missing options
    # -------------------------------------------------------------
    # Helper: numeric menu selection
    def numeric_choice(prompt_text: str, options: list[str], default_idx: int = 0) -> str:
        click.echo(prompt_text)
        for idx, option in enumerate(options, start=1):
            click.echo(f"  {idx}) {option}")
        choice_num = click.prompt("Enter choice number", type=int, default=default_idx + 1)
        if 1 <= choice_num <= len(options):
            return options[choice_num - 1]
        click.echo("Invalid choice, using default.")
        return options[default_idx]

    if mode is None:
        mode = numeric_choice("Select mode:", ["live", "offline", "eval"], default_idx=0)

    if source is None:
        if mode == "live":
            source = click.prompt("Enter camera index (integer) or RTSP URL", default="0")
        else:
            source = click.prompt("Enter path to video file or directory", default="videos")

    if not count_line:
        if click.confirm("Would you like to define counting lines?", default=False):
            collected = []
            while True:
                entry = click.prompt("Enter counting line as x1,y1,x2,y2 or 'done' when finished", default="done")
                if entry.lower() in ("done", "none"):
                    break
                collected.append(entry)
            count_line = tuple(collected)  # convert to tuple to mimic Click multi option structure

    # -------------------------------------------------------------
    # Validate mode/source combination and apply overrides
    # -------------------------------------------------------------
    # ------------------------------------------------------------------
    # Configure universal output directory → each run gets its own subfolder
    # under data/videos/output/<TIMESTAMP>/ so files stay grouped together.
    # ------------------------------------------------------------------
    base_output_dir = Path("data/videos/output")

    # Create base folder if it doesn't exist
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped subfolder for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_output_dir = base_output_dir / timestamp
    try:
        session_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create session output directory {session_output_dir}: {e}")

    if mode == "live":
        # Determine if source is an int (camera index) or string (RTSP/url)
        try:
            video_source = int(source)
        except ValueError:
            video_source = source  # treat as RTSP/url

        # By default, live mode does NOT record video; user can override via config
        config_data.setdefault("visualizer", {}).setdefault("save_to_file", False)
    else:  # offline mode
        path_obj = Path(source)
        if not path_obj.exists():
            logger.error("Offline mode requires a valid file or directory path in --source")
            sys.exit(1)
        video_source = str(path_obj)

        # Offline mode should record processed video
        config_data.setdefault("visualizer", {})["save_to_file"] = True

    # Tell visualizer and summary service where to store outputs
    config_data.setdefault("visualizer", {})["save_path"] = str(session_output_dir)
    config_data.setdefault("summary_service", {})["summary_output_dir"] = str(session_output_dir)
 
    # For eval mode, if start/max not provided, prompt quickly
    if mode == "eval":
        if start_sec is None:
            start_sec = click.prompt("Start time in seconds", type=float, default=0.0)
        if max_frames is None:
            max_frames = click.prompt("Max frames to process", type=int, default=150)

    # -------------------------------------------------------------
    # Ensure frame grabber uses the chosen video source for ALL modes
    # -------------------------------------------------------------
    config_data.setdefault("frame_grabber", {})["video_source"] = video_source
    # Pass evaluation-mode limits if provided
    if start_sec is not None:
        config_data["frame_grabber"]["start_time_sec"] = start_sec
    if max_frames is not None:
        config_data["frame_grabber"]["max_frames"] = max_frames

    # -------------------------------------------------------------
    # Counting line handling
    # -------------------------------------------------------------
    if count_line:
        if len(count_line) == 1 and count_line[0].lower() == "none":
            config_data.setdefault("vehicle_counter", {})["counting_lines"] = []
        else:
            lines_list = []
            for line_str in count_line:
                try:
                    parts = [float(p) for p in line_str.split(",") if p.strip() != ""]
                    if len(parts) != 4:
                        raise ValueError
                    # Convert to nested list [[x1,y1],[x2,y2]]
                    nested = [[parts[0], parts[1]], [parts[2], parts[3]]]
                    # If all values are integers (no decimal), cast to int for absolute coords convenience
                    if all(abs(p - int(p)) < 1e-6 for p in parts):
                        nested = [[int(parts[0]), int(parts[1])], [int(parts[2]), int(parts[3])]]
                    lines_list.append(nested)
                except ValueError:
                    logger.error(f"Invalid --count-line format: '{line_str}'. Expected x1,y1,x2,y2")
                    sys.exit(1)
            config_data.setdefault("vehicle_counter", {})["counting_lines"] = lines_list

    # -------------------------------------------------------------
    # Merge CLI overrides with base YAML (default settings or provided via --config)
    # -------------------------------------------------------------
    from traffic_monitor.utils.config_loader import load_config as _tm_load_cfg
    from copy import deepcopy
    from pathlib import Path as _Path

    def _deep_update(dest: dict, src: dict):
        """Recursively update dict dest with src (src overrides)."""
        for key, val in src.items():
            if isinstance(val, dict) and isinstance(dest.get(key), dict):
                _deep_update(dest[key], val)
            else:
                dest[key] = val
        return dest

    # If a --config file was provided, config_data is already populated.
    # If not, config_data contains only interactive overrides.
    # We will pass this directly to the supervisor, which will handle merging
    # with the default settings.yaml.
    final_cfg = config_data
    logger.debug(f"CLI passing config to supervisor: {final_cfg}")

    # -------------------------------------------------------------
    # Run supervisor with merged configuration
    # -------------------------------------------------------------
    supervisor_main(config=final_cfg)


if __name__ == "__main__":
    main() 
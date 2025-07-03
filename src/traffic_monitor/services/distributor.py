import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import List
from loguru import logger
from ..utils.queue_utils import safe_put

def distributor_process(
    offline_mode: bool,
    input_queue: Queue,
    output_queues: List[Queue], # It takes a LIST of output queues
    shutdown_event: Event
):
    process_name = mp.current_process().name
    logger.info(f"[{process_name}] Distributor process started.")

    try:
        while not shutdown_event.is_set():
            try:
                message = input_queue.get(timeout=1.0)
            except Empty:
                continue

            if message is None:
                # When the input is done, propagate the shutdown signal to all outputs
                logger.info(f"[{process_name}] Received None sentinel. Propagating to all output queues.")
                for q in output_queues:
                    q.put(None)
                break

            # Distribute message to all output queues using mode-aware operations
            for i, q in enumerate(output_queues):
                success = safe_put(q, message, offline_mode, f"{process_name}-Branch{i}")
                if not success:
                    logger.warning(f"[{process_name}] Failed to put message to output queue {i}")
    
    except Exception:
        logger.exception(f"[{process_name}] An unhandled error occurred.")
    finally:
        logger.info(f"[{process_name}] Distributor process shutting down.")
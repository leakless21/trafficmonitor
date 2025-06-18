import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import List
from loguru import logger

def distributor_process(
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

            # For each output queue, put a copy of the message.
            for q in output_queues:
                try:
                    q.put(message, timeout=1.0)
                except Full:
                    logger.warning(f"[{process_name}] An output queue is full. Message may be dropped for that branch.")
    
    except Exception:
        logger.exception(f"[{process_name}] An unhandled error occurred.")
    finally:
        logger.info(f"[{process_name}] Distributor process shutting down.")
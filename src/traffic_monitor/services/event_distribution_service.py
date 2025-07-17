import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty
from typing import List
from loguru import logger
from ..utils.queue_utils import safe_put

def event_distribution_process(
    offline_mode: bool,
    input_queue: Queue,
    output_queues: List[Queue], # It takes a LIST of output queues
    shutdown_event: Event
):
    process_name = mp.current_process().name
    logger.info(f"[{process_name}] Distributor process started with {len(output_queues)} output queues.")

    message_count = 0
    try:
        while not shutdown_event.is_set():
            try:
                message = input_queue.get(timeout=1.0)
            except Empty:
                continue

            if message is None:
                # When the input is done, propagate the shutdown signal to all outputs
                logger.info(f"[{process_name}] Received None sentinel after {message_count} messages. Propagating to all output queues.")
                for i, q in enumerate(output_queues):
                    q.put(None)
                    logger.debug(f"[{process_name}] Sent None to output queue {i}")
                break

            message_count += 1
            
            # Log message details for debugging
            if message_count <= 5 or message_count % 10 == 0:
                msg_type = type(message).__name__ if hasattr(message, '__class__') else str(type(message))
                frame_id = message.get('frame_id', 'unknown') if isinstance(message, dict) else 'non-dict'
                logger.info(f"[{process_name}] Processing message {message_count}: type={msg_type}, frame_id={frame_id}")

            # Distribute message to all output queues using mode-aware operations
            success_count = 0
            for i, q in enumerate(output_queues):
                success = safe_put(q, message, offline_mode, f"{process_name}-Branch{i}")
                if success:
                    success_count += 1
                    # Special logging for fusion queues
                    if 'fusion' in process_name.lower():
                        logger.debug(f"[{process_name}] Successfully sent message {message_count} to fusion queue {i}")
                else:
                    logger.warning(f"[{process_name}] Failed to put message {message_count} to output queue {i}")
            
            if success_count != len(output_queues):
                logger.warning(f"[{process_name}] Only {success_count}/{len(output_queues)} queues received message {message_count}")
    
    except Exception:
        logger.exception(f"[{process_name}] An unhandled error occurred after {message_count} messages.")
    finally:
        logger.info(f"[{process_name}] Distributor process shutting down after processing {message_count} messages.")
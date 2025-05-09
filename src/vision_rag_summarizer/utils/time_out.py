

import asyncio
import logging

async def run_with_timeout(func, *args, timeout=60):
    """
    Runs a blocking function with a timeout in an executor.

    :param func: Function to run
    :param args: Arguments to pass to the function
    :param timeout: Timeout in seconds
    :return: Result of the function or timeout placeholder
    """
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, func, *args),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logging.error(f"⏱️ Timeout after {timeout} seconds")
        return "[Timeout]"
    except Exception as e:
        logging.error(f"❌ Exception in run_with_timeout: {e}")
        return "[Error]"

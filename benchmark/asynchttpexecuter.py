
import aiohttp
import asyncio
import time
import signal
import logging
import os
from datetime import timedelta
from typing import Callable
from .ratelimiting import NoRateLimiter

# Threshold in seconds to warn about requests lagging behind target rate.
LAG_WARN_DURATION = 1.0

class AsyncHTTPExecuter:
    """
    An implementation of an async HTTP executer class with rate limiting and
    concurrency control.
    """
    def __init__(self, async_http_func: Callable[[aiohttp.ClientSession], None], rate_limiter=NoRateLimiter(), max_concurrency=12, finish_run_func=None):
        """
        Creates a new executer.
        :param async_http_func: A callable function that takes aiohttp.ClientSession to use to perform request.
        :param rate_limiter: Rate limiter object to use, defaults to NoRateLimiter.
        :param max_concurrency: Maximum number of concurrent requests, defaults to 12.
        :param finish_run_func: Function to run when run reaches end.
        """
        self.async_http_func = async_http_func
        self.rate_limiter = rate_limiter
        self.max_concurrency = max_concurrency
        self.max_lag_warn = timedelta(seconds=5).seconds
        self.terminate = False
        self.finish_run_func = finish_run_func

    def run(self, call_count=None, duration=None, run_end_condition_mode="or"):
        """
        Runs the executer. If call_count and duration not specified, it will run until cancelled.
        :param call_count: Number of calls to execute, default infinite.
        :param duration: Duration in second for the run, default infinite.
        :param run_end_condition_mode: Criteria to use to determine when to stop the run. "and" will stop when both call_count and duration are reached, "or" will stop when either call_count or duration is reached. Defaults to "or"
        """
        asyncio.run(self._run(call_count=call_count, duration=duration, run_end_condition_mode=run_end_condition_mode))

    async def _run(self, call_count=None, duration=None, run_end_condition_mode="or"):
        orig_sigint_handler = signal.signal(signal.SIGINT, self._terminate)
        orig_sigterm_handler = signal.signal(signal.SIGTERM, self._terminate)
        # disable all TCP limits for highly parallel loads
        conn = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=conn) as session:
            start_time = time.time()
            calls_made = 0
            request_tasks = set()
            run_end_conditions_met = False
            while not run_end_conditions_met and not self.terminate:
                async with self.rate_limiter:
                    if len(request_tasks) > self.max_concurrency:
                        wait_start_time = time.time()
                        _, crs_pending = await asyncio.wait(request_tasks, return_when=asyncio.FIRST_COMPLETED)
                        request_tasks = crs_pending
                        waited = time.time() - wait_start_time
                        if waited > LAG_WARN_DURATION and type(self.rate_limiter) is not NoRateLimiter:
                            logging.warning(f"falling behind committed rate by {round(waited, 3)}s, consider increasing number of clients.")
                    v = asyncio.create_task(self.async_http_func(session))
                    request_tasks.add(v)
                    calls_made += 1
                    # Determine whether to end the run
                    if call_count is None and duration is None:
                        run_end_conditions_met = False
                    elif run_end_condition_mode == "and":
                        request_limit_reached = call_count is None or calls_made >= call_count
                        duration_limit_reached = duration is None or (time.time() - start_time) > duration
                        run_end_conditions_met = request_limit_reached and duration_limit_reached
                    else: # "or"
                        request_limit_reached = call_count is not None and calls_made >= call_count
                        duration_limit_reached = duration is not None and (time.time() - start_time) > duration
                        run_end_conditions_met = request_limit_reached or duration_limit_reached

            if len(request_tasks) > 0:
                logging.info(f"waiting for {len(request_tasks)} requests to drain")
                await asyncio.wait(request_tasks)

            if self.finish_run_func:
                self.finish_run_func()

        signal.signal(signal.SIGINT, orig_sigint_handler)
        signal.signal(signal.SIGTERM, orig_sigterm_handler)

    def _terminate(self, *args):
        if not self.terminate:
            logging.warning("got terminate signal, draining. signal again to exit immediately.")
            self.terminate = True
        else:
            logging.info("forcing program exit")
            os._exit(0)

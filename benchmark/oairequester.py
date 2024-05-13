# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import logging
import time
from typing import Optional
import json

import aiohttp
import backoff

# TODO: switch to using OpenAI client library once new headers are exposed.

REQUEST_ID_HEADER = "apim-request-id"
UTILIZATION_HEADER = "azure-openai-deployment-utilization"
RETRY_AFTER_MS_HEADER = "retry-after-ms"
MAX_RETRY_SECONDS = 60.0

TELEMETRY_USER_AGENT_HEADER = "x-ms-useragent"
USER_AGENT = "aoai-benchmark"

class RequestStats:
    """
    Statistics collected for a particular AOAI request.
    """
    def __init__(self):
        self.request_start_time: Optional[float] = None
        self.response_status_code: int = 0
        self.response_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.response_end_time: Optional[float] = None
        self.context_tokens: int = 0
        self.generated_tokens: Optional[int] = None
        self.deployment_utilization: Optional[float] = None
        self.calls: int = 0
        self.last_exception: Optional[Exception] = None
        self.input_messages: Optional[dict[str, str]] = None
        self.output_content: list[dict] = list()

    def as_dict(self, include_request_content: bool = False) -> dict:
        output = {
            "request_start_time": self.request_start_time,
            "response_status_code": self.response_status_code,
            "response_time": round(self.response_time, 4),
            "first_token_time": round(self.first_token_time, 4),
            "response_end_time": round(self.response_end_time, 4),
            "context_tokens": self.context_tokens,
            "generated_tokens": self.generated_tokens,
            "deployment_utilization": self.deployment_utilization,
            "calls": self.calls,
            "last_exception": self.last_exception,
        }
        if include_request_content:
            output["input_messages"] = self.input_messages
            output["output_content"] = self.output_content if self.output_content else None
        return output

def _terminal_http_code(e) -> bool:
    # we only retry on 429
    return e.response.status != 429

class OAIRequester:
    """
    A simple AOAI requester that makes a streaming call and collect corresponding
    statistics.
    :param api_key: Azure OpenAI resource endpoint key.
    :param url: Full deployment URL in the form of https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completins?api-version=<api_version>
    :param backoff: Whether to retry throttled or unsuccessful requests.
    """
    def __init__(self, api_key: str, url: str, backoff=False):
        self.api_key = api_key
        self.url = url
        self.backoff = backoff

    async def call(self, session:aiohttp.ClientSession, body: dict) -> RequestStats:
        """
        Makes a single call with body and returns statistics. The function
        forces the request in streaming mode to be able to collect token
        generation latency.
        In case of failure, if the status code is 429 due to throttling, value
        of header retry-after-ms will be honored. Otherwise, request
        will be retried with an exponential backoff.
        Any other non-200 status code will fail immediately.

        :param body: json request body.
        :return RequestStats.
        """
        stats = RequestStats()
        stats.input_messages = body["messages"]
        # operate only in streaming mode so we can collect token stats.
        body["stream"] = True
        try:
            await self._call(session, body, stats)
        except Exception as e:
            stats.last_exception = e

        return stats

    @backoff.on_exception(backoff.expo,
                      aiohttp.ClientError,
                      jitter=backoff.full_jitter,
                      max_time=MAX_RETRY_SECONDS,
                      giveup=_terminal_http_code)
    async def _call(self, session:aiohttp.ClientSession, body: dict, stats: RequestStats):
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
            TELEMETRY_USER_AGENT_HEADER: USER_AGENT,
        }
        stats.request_start_time = time.time()
        while stats.calls == 0 or time.time() - stats.request_start_time < MAX_RETRY_SECONDS:
            stats.calls += 1
            response = await session.post(self.url, headers=headers, json=body)
            stats.response_status_code = response.status
            # capture utilization in all cases, if found
            self._read_utilization(response, stats)
            if response.status != 429:
                break
            if self.backoff and RETRY_AFTER_MS_HEADER in response.headers:
                try:
                    retry_after_str = response.headers[RETRY_AFTER_MS_HEADER]
                    retry_after_ms = float(retry_after_str)
                    logging.debug(f"retry-after sleeping for {retry_after_ms}ms")
                    await asyncio.sleep(retry_after_ms/1000.0)
                except ValueError as e:
                    logging.warning(f"unable to parse retry-after header value: {UTILIZATION_HEADER}={retry_after_str}: {e}")   
                    # fallback to backoff
                    break
            else:
                # fallback to backoff
                break

        if response.status != 200:
            stats.response_end_time = time.time()
        if response.status != 200 and response.status != 429:
            logging.warning(f"call failed: {REQUEST_ID_HEADER}={response.headers[REQUEST_ID_HEADER]} {response.status}: {response.reason}")
        if self.backoff:
            response.raise_for_status()
        if response.status == 200:
            await self._handle_response(response, stats)
        
    async def _handle_response(self, response: aiohttp.ClientResponse, stats: RequestStats):
        async with response:
            stats.response_time = time.time()
            async for line in response.content:
                if not line.startswith(b'data:'):
                    continue
                if stats.first_token_time is None:
                    stats.first_token_time = time.time()
                if stats.generated_tokens is None:
                    stats.generated_tokens = 0
                # Save content from generated tokens
                content = line.decode('utf-8')
                if content == "data: [DONE]\n":
                    # Request is finished - no more tokens to process
                    break
                content = json.loads(content.replace("data: ", ""))["choices"][0]["delta"]
                if content:
                    if next(iter(content)) == "role":
                        stats.output_content.append({"role": content["role"], "content": ""})
                    else:
                        stats.output_content[-1]["content"] += content["content"]
                    stats.generated_tokens += 1
            stats.response_end_time = time.time()

    def _read_utilization(self, response: aiohttp.ClientResponse, stats: RequestStats):
        if UTILIZATION_HEADER in response.headers:
            util_str = response.headers[UTILIZATION_HEADER]
            if len(util_str) == 0:
                logging.warning(f"got empty utilization header {UTILIZATION_HEADER}")
            elif util_str[-1] != '%':
                logging.warning(f"invalid utilization header value: {UTILIZATION_HEADER}={util_str}")
            else:
                try:
                    stats.deployment_utilization = float(util_str[:-1])
                except ValueError as e:
                    logging.warning(f"unable to parse utilization header value: {UTILIZATION_HEADER}={util_str}: {e}")            


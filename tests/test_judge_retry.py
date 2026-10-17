import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from codi_repro.judge import complete


class JudgeRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_response_body_is_retried(self):
        requested = []
        def reply(request):
            requested.append(request)
            if len(requested) == 1: raise httpx.ReadError('connection closed', request=request)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]})
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('codi_repro.judge.asyncio.sleep', new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
                result = await complete(client, asyncio.Semaphore(1), 'test', 'test', 128, {'type': 'json_object'})
        self.assertEqual(result, '{}')
        self.assertEqual(len(requested), 2)

    async def test_incomplete_output_is_retried_with_a_larger_budget(self):
        responses = [{'choices': [{'finish_reason': 'length', 'message': {'content': '{'}}], 'usage': {'completion_tokens': 128}}, {'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}]
        budgets = []
        def reply(request):
            import json
            budgets.append(json.loads(request.content)['max_tokens'])
            return httpx.Response(200, json=responses[len(budgets) - 1])
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('codi_repro.judge.asyncio.sleep', new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
                result = await complete(client, asyncio.Semaphore(1), 'test', 'test', 128, {'type': 'json_object'})
        self.assertEqual(result, '{}')
        self.assertEqual(budgets, [128, 256])

    async def test_upstream_rate_limit_inside_http_success_is_retried(self):
        responses = [{'choices': [{'finish_reason': 'error', 'error': {'code': 429, 'message': 'upstream busy'}}]}, {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"values":[]}'}}]}]
        requested = []
        def reply(request):
            requested.append(request)
            return httpx.Response(200, json=responses[len(requested) - 1])
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('codi_repro.judge.asyncio.sleep', new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
                result = await complete(client, asyncio.Semaphore(1), 'test', 'test', 128, {'type': 'json_object'})
        self.assertEqual(result, '{"values":[]}')
        self.assertEqual(len(requested), 2)

    async def test_authentication_failure_is_not_retried(self):
        requests = []
        def reply(request):
            requests.append(request)
            return httpx.Response(401, json={'error': 'invalid credentials'})
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
                with self.assertRaises(httpx.HTTPStatusError):
                    await complete(client, asyncio.Semaphore(1), 'test', 'test', 128, {'type': 'json_object'})
        self.assertEqual(len(requests), 1)


if __name__ == '__main__': unittest.main()

from __future__ import annotations

import asyncio
import sys
import types
import unittest

from ru_local_avatar_agent.domain.session import SessionStateMachine
from ru_local_avatar_agent.voice.runtime import VoiceRuntime


class _Worker:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


async def _run_until_closed(worker: _Worker) -> None:
    while not worker.closed:
        await asyncio.sleep(0.01)


class VoiceRuntimeWorkerTest(unittest.TestCase):
    def test_stop_worker_closes_task_and_removes_maps(self) -> None:
        async def scenario() -> None:
            runtime = VoiceRuntime.__new__(VoiceRuntime)
            worker = _Worker()
            task = asyncio.create_task(_run_until_closed(worker))
            runtime._workers = {"session-1": worker}
            runtime._worker_tasks = {"session-1": task}

            stopped = await VoiceRuntime.stop_worker(
                runtime, "session-1", timeout_seconds=1.0
            )

            self.assertTrue(stopped)
            self.assertTrue(worker.closed)
            self.assertTrue(task.done())
            self.assertEqual(runtime._workers, {})
            self.assertEqual(runtime._worker_tasks, {})

        asyncio.run(scenario())

    def test_worker_finish_calls_release_callback(self) -> None:
        class CompletingWorker:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            async def run(self) -> None:
                return

        async def scenario() -> None:
            runtime = VoiceRuntime.__new__(VoiceRuntime)
            runtime._workers = {}
            runtime._worker_tasks = {}
            released: list[str] = []
            session = SessionStateMachine("session-1")
            module_name = "ru_local_avatar_agent.voice.worker"
            previous_module = sys.modules.get(module_name)
            fake_module = types.ModuleType(module_name)
            fake_module.VoiceSessionWorker = CompletingWorker
            sys.modules[module_name] = fake_module

            try:
                started = VoiceRuntime.start_worker(
                    runtime,
                    session,
                    "room-1",
                    on_stop=released.append,
                )
                await runtime._worker_tasks["session-1"]
            finally:
                if previous_module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = previous_module

            self.assertTrue(started)
            self.assertEqual(released, ["session-1"])
            self.assertEqual(runtime._workers, {})
            self.assertEqual(runtime._worker_tasks, {})

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

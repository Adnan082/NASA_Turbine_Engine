import asyncio
from collections import defaultdict
from typing import Callable, Any


class EventBus:
    def __init__(self):
        self._subscribers = defaultdict(list)
        self._results     = {}

    def subscribe(self, event_name: str, handler: Callable):
        self._subscribers[event_name].append(handler)

    async def publish(self, event_name: str, event_data: Any):
        handlers = self._subscribers.get(event_name, [])
        for handler in handlers:
            await handler(event_data)

    def store_result(self, engine_id: int, result: dict):
        self._results[engine_id] = result

    def get_result(self, engine_id: int):
        return self._results.get(engine_id, None)

    def get_all_results(self):
        return self._results

    def clear(self):
        self._results = {}


# global bus instance
bus = EventBus()

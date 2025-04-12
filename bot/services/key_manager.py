import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from bot.config import OpenRouter_API_KEYS


class OpenRouterKeyManager:
    def __init__(self):
        self.keys = OpenRouter_API_KEYS
        self.usage = defaultdict(int)
        self.last_reset = datetime.now()
        self.failed_keys = set()
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(5)

    async def get_key(self):
        async with self.lock:
            if datetime.now() - self.last_reset >= timedelta(hours=24):
                self.usage.clear()
                self.failed_keys.clear()
                self.last_reset = datetime.now()

            for key in self.keys:
                if key not in self.failed_keys and self.usage[key] < 45:
                    return key

            for key in self.keys:
                if key not in self.failed_keys:
                    return key

            return None

    async def update_usage(self, key, response):
        async with self.lock:
            if response.status == 200:
                self.usage[key] += 1
            elif response.status == 429:
                self.failed_keys.add(key)

    async def handle_error(self, key):
        async with self.lock:
            self.failed_keys.add(key)

    async def get_key_statuses(self):
        status = []
        for key in self.keys:
            status.append({
                "key": f"{key[:5]}...{key[-5:]}",
                "used": self.usage.get(key, 0),
                "status": "active" if key not in self.failed_keys else "expired"
            })
        return status



key_manager = OpenRouterKeyManager()
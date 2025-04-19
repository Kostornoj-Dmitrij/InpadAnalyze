import os
from aiogram import Bot
import asyncio
import uuid
import base64

async def cleanup_temp_files(*paths):
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Error deleting file {path}: {e}")


async def send_long_message(bot: Bot, chat_id: int, text: str, chunk_size: int = 4096, delay: float = 0.3) -> None:
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]

        if i + chunk_size < len(text) and '\n' in chunk:
            last_newline = chunk.rfind('\n')
            if last_newline > 0:
                chunk = chunk[:last_newline + 1]
                i = i - (chunk_size - last_newline - 1)

        await bot.send_message(chat_id, chunk)
        await asyncio.sleep(delay)

async def generate_short_uuid() -> str:
    """Асинхронно генерирует короткий (6 символов) UUID"""
    uuid_bytes = uuid.uuid4().bytes
    encoded = base64.urlsafe_b64encode(uuid_bytes).decode('utf-8').rstrip('=')
    return encoded[:6]
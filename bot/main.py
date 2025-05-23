from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from handlers import start, comparison, questions, common
from config import TOKEN
import logging

"""
Для хостинга
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(),
    ]
)

async def main():
    '''
    session = AiohttpSession(
        api=TelegramAPIServer.from_base("http://localhost:8081", is_local=True),
        timeout=60 * 30
    )
    '''

    storage = MemoryStorage()
    bot = Bot(token=TOKEN)
    dp = Dispatcher(storage=storage)

    dp.include_routers(
        start.router,
        comparison.router,
        questions.router,
        common.router
    )

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
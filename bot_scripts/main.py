from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiosqlite import connect

from config import TOKEN
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),  # запись в файл
        logging.StreamHandler(),         # вывод в консоль
    ]
)

session = AiohttpSession(
    api=TelegramAPIServer.from_base("http://localhost:8081", is_local=True),
    timeout=60*30
)

bot = Bot(token = TOKEN, session=session)
dp = Dispatcher()

user_data = {}
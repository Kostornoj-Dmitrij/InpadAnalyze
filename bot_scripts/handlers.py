import os
import logging
import time

from config import TOKEN, GigaChat_API_KEY
from user import User
from main import bot, dp, user_data
import utils
import kb
import asyncio
from aiogram import types
import requests
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command, CommandStart
from aiogram.fsm.storage.memory import MemoryStorage

# Логирование
logging.basicConfig(level=logging.INFO)

# Определение состояний
class FileStates(StatesGroup):
    waiting_for_tz = State()
    waiting_for_result = State()
    question_input = State()


# Обработчик команды /start
@dp.message(CommandStart())
async def start(message: types.Message):
    user_id = message.chat.id
    user = User(user_id)
    user_data[user_id] = user
    await utils.user_clear(message)
    keyboard = kb.start_keyboard
    await message.answer("Привет! Добро пожаловать в бота.", reply_markup=keyboard)

@dp.message(Command("start"))
async def start(message: types.Message):
    user_id = message.chat.id
    user = User(user_id)
    user_data[user_id] = user
    await utils.user_clear(message)
    keyboard = kb.start_keyboard
    await message.answer("Привет! Добро пожаловать в бота.", reply_markup=keyboard)

# Обработчик загрузки файла ТЗ
@dp.message(FileStates.waiting_for_tz, lambda message: message.content_type == "document")
async def handle_tz_file(message: types.Message, state: FSMContext):
    user_id = message.chat.id
    user = user_data[user_id]
    logging.info("Сохранение файла")

    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Сохраняем файл
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    logging.info("Бот получил файл")
    file_path = file.file_path
    downloaded_file = await bot.download_file(file_path)
    logging.info("Бот скачал файл")
    user.tz_file_path = f"temp/{user_id}_tz.pdf"
    with open(user.tz_file_path, "wb") as new_file:
        new_file.write(downloaded_file.read())

    if user_data[user_id].state == 'question_input':
        await message.answer("Файл успешно загружен, теперь введите ваш вопрос.")
        await state.set_state(FileStates.question_input)
    else:
        await message.answer("Файл ТЗ успешно загружен. Теперь загрузите файл с результатом.")
        await state.set_state(FileStates.waiting_for_result)

# Обработчик загрузки файла результата
@dp.message(FileStates.waiting_for_result, lambda message: message.content_type == "document")
async def handle_result_file(message: types.Message, state: FSMContext):
    user_id = message.chat.id
    user = user_data[user_id]

    # Сохраняем файл
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    downloaded_file = await bot.download_file(file_path)

    user.result_file_path = f"temp/{user_id}_result.pdf"
    with open(user.result_file_path, "wb") as new_file:
        new_file.write(downloaded_file.read())

    await message.answer("Файл с результатом успешно загружен. Начинаю анализ...")
    await utils.analyze_files(user_id)

    await state.set_state(None)
    await start(message)

@dp.message(FileStates.question_input)
async def handle_result_file(message: types.Message, state: FSMContext):
    user_id = message.chat.id
    user = user_data[user_id]

    await message.answer("Ответ...")
    await start(message)

# Обработчик текстовых сообщений
@dp.message(lambda message: user_data[message.chat.id].state == 'asking_question')
async def asking_question(message: types.Message):
    await utils.answer_generation(message)
    keyboard = kb.question_keyboard
    await message.answer("Вам помог ответ?", reply_markup=keyboard)

# Обработчик команды /question_start
@dp.message(Command("question_start"))
async def handle_question_start(message: types.Message):
    user_id = message.chat.id
    await utils.show_buildings_options(user_id)

# Обработчик callback-запросов
@dp.callback_query()
async def callback_inline(call: types.CallbackQuery, state: FSMContext):
    user_id = call.message.chat.id

    if call.data == 'start':
        await bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        await start(call.message)
    elif call.data == 'question_start':
        await bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        await utils.show_buildings_options(call.message.chat.id)
    elif call.data == 'brusnika':
        await bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        await utils.show_questions_options(call.message.chat.id)
    elif call.data == 'analyze_docs':
        await bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        keyboard = kb.categories_keyboard
        await call.message.answer("Выберите, по какому разделу будем сравнивать.", reply_markup=keyboard)
    elif call.data == 'arch_category':
        await state.set_state(FileStates.waiting_for_tz)
        await bot.send_message(user_id, "Пожалуйста, загрузите файл с ТЗ.")
    elif call.data == 'constr_category':
        await state.set_state(FileStates.waiting_for_tz)
        await bot.send_message(user_id, "Пожалуйста, загрузите файл с ТЗ.")
    elif call.data == 'eng_category':
        await state.set_state(FileStates.waiting_for_tz)
        await bot.send_message(user_id, "Пожалуйста, загрузите файл с ТЗ.")
    elif call.data == 'question':
        await bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        user_data[call.message.chat.id].state = 'question_input'
        await state.set_state(FileStates.waiting_for_tz)
        await bot.send_message(user_id, "Пожалуйста, загрузите файл, по которому хотите задать вопрос.")
    else:
        await call.answer("Ошибка!")

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
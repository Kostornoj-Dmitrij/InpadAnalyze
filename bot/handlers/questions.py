from aiogram import Router, types, F
from aiogram.fsm.context import FSMContext

from handlers.start import cmd_start
from models.states import QuestionStates
from services.file_processing import process_pdf_file
from services.analysis import generate_answer
import os
from services.utils import send_long_message
import asyncio
from aiogram import Bot

router = Router()


@router.callback_query(F.data == 'ask_question')
async def start_question_flow(callback: types.CallbackQuery, state: FSMContext):
    try:
        await callback.message.delete()
    except:
        pass
    await callback.message.answer("Пожалуйста, загрузите PDF-файл для анализа:")
    await state.set_state(QuestionStates.waiting_file)



@router.message(
    QuestionStates.waiting_file,
    F.document & (F.document.mime_type == 'application/pdf')
)
async def handle_question_file(message: types.Message, state: FSMContext, bot: Bot):
    file_path = await process_pdf_file(message, message.bot)
    await state.update_data(file_path=file_path)

    await asyncio.sleep(5)
    user_data = await state.get_data()

    await message.answer("Файл получен. Теперь задайте ваш вопрос:")
    await state.set_state(QuestionStates.waiting_question)


@router.message(QuestionStates.waiting_question)
async def handle_question(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    await message.answer("Ищу ответ, ожидайте пожалуйста.")


    try:
        answer = await generate_answer(user_data['file_path'], message.text)
        await send_long_message(message.bot, message.chat.id, answer)

    except Exception as e:
        error_msg = f"Ошибка: {str(e)[:300]}"
        await message.answer(error_msg)

    finally:
        if 'file_path' in user_data and os.path.exists(user_data['file_path']):
            os.remove(user_data['file_path'])
        await state.clear()
    await cmd_start(message, state)
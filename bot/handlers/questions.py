from aiogram import Router, types, F
from aiogram.fsm.context import FSMContext

from bot.handlers.start import cmd_start
from bot.models.states import QuestionStates
from bot.services.file_processing import process_pdf_file
from bot.services.analysis import generate_answer
import os

router = Router()


@router.callback_query(F.data == 'ask_question')
async def start_question_flow(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.answer("Пожалуйста, загрузите PDF-файл для анализа:")
    await state.set_state(QuestionStates.waiting_file)


@router.message(
    QuestionStates.waiting_file,
    F.document & (F.document.mime_type == 'application/pdf')
)
async def handle_question_file(message: types.Message, state: FSMContext):
    file_path = await process_pdf_file(message, message.bot)
    await state.update_data(file_path=file_path)
    await message.answer("Файл получен. Теперь задайте ваш вопрос:")
    await state.set_state(QuestionStates.waiting_question)


@router.message(QuestionStates.waiting_question)
async def handle_question(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    answer = await generate_answer(user_data['file_path'], message.text)

    await message.answer(answer)
    if os.path.exists(user_data['file_path']):
        os.remove(user_data['file_path'])
    await state.clear()
    await cmd_start(message, state)
from aiogram import Router, types
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from bot.keyboards.builders import KeyboardBuilder
from bot.services.file_processing import cleanup_temp_files


router = Router()

@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    user_data = await state.get_data()

    files_to_clean = [
        user_data.get('tz_file'),
        user_data.get('result_path'),
        user_data.get('report_path')
    ]

    await cleanup_temp_files([f for f in files_to_clean if f is not None])

    await state.clear()
    await message.answer(
        "Привет! Добро пожаловать в бота.", reply_markup=KeyboardBuilder.start_kb()
    )
from aiogram import Router, types, F
from aiogram.fsm.context import FSMContext
from aiogram.types import FSInputFile

from bot.models.states import ComparisonStates
from bot.keyboards.builders import KeyboardBuilder
from bot.services.file_processing import process_pdf_file, cleanup_temp_files
from bot.services.analysis import analyze_documents
from bot.handlers.start import cmd_start
import os

router = Router()


@router.callback_query(F.data == 'analyze_docs')
async def choose_category(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text(
        "Выберите раздел для сравнения:",
        reply_markup=KeyboardBuilder.categories_kb()
    )
    await state.set_state(ComparisonStates.choosing_category)

@router.callback_query(F.data == 'eng_options')
async def choose_category(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text(
        "Выберите конкретный раздел инженерных решений:",
        reply_markup=KeyboardBuilder.eng_options_kb()
    )
    await state.set_state(ComparisonStates.choosing_category)

@router.callback_query(
    ComparisonStates.choosing_category,
    F.data.in_(['arch_category', 'constr_category', 'eng_category', 'water_supply_category', 'water_drain_category', 'heat_network_category', 'hvac_category'])
)
async def handle_category_selection(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(category=callback.data)
    await callback.message.answer("Пожалуйста, загрузите файл с ТЗ:")
    await state.set_state(ComparisonStates.waiting_tz_file)


@router.message(
    ComparisonStates.waiting_tz_file,
    F.document & (F.document.mime_type == 'application/pdf')
)
async def handle_tz_file(message: types.Message, state: FSMContext):
    file_path = await process_pdf_file(message, message.bot)
    await state.update_data(tz_file=file_path)
    await message.answer("Файл ТЗ успешно загружен. Теперь загрузите файл с результатом:")
    await state.set_state(ComparisonStates.waiting_result_file)


@router.message(ComparisonStates.waiting_result_file)
async def handle_result_file(message: types.Message, state: FSMContext):
    user_data = await state.get_data()

    result_path = await process_pdf_file(message, message.bot)

    await state.update_data(result_path=result_path)

    category = user_data.get('category')

    await message.answer("Начинаю анализ... Ожидайте пожалуйста.")
    try:
        report_path = await analyze_documents(
            category=category,
            tz_path=user_data['tz_file'],
            result_path=result_path,
            user_id=message.from_user.id
        )

        await message.answer_document(
            FSInputFile(report_path),
            caption="Отчет готов"
        )
    except Exception as e:
        await message.answer(f"Произошла ошибка: {str(e)}")
        await cmd_start(message, state)
    finally:
        files_to_clean = [
            user_data.get('tz_file'),
            result_path,
            report_path if 'report_path' in locals() else None
        ]
        await cleanup_temp_files([f for f in files_to_clean if f is not None])
        await state.clear()
        await cmd_start(message, state)
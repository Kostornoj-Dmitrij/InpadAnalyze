from aiogram import Router, types, F
from aiogram.fsm.context import FSMContext
from bot.models.states import ComparisonStates
from bot.keyboards.builders import KeyboardBuilder
from bot.services.file_processing import process_pdf_file
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


@router.callback_query(
    ComparisonStates.choosing_category,
    F.data.in_(['arch_category', 'constr_category', 'eng_category'])
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


@router.message(
    ComparisonStates.waiting_result_file,
    F.document & (F.document.mime_type == 'application/pdf')
)
async def handle_result_file(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    result_file = await process_pdf_file(message, message.bot)
    report_path = ""
    await message.answer("Начинаю анализ документов...")
    try:
        report_path = await analyze_documents(
            user_data['tz_file'],
            result_file,
            message.from_user.id
        )
        await message.answer_document(
            types.FSInputFile(report_path),
            caption="✅ Отчёт готов"
        )
    finally:
        for path in [user_data['tz_file'], result_file, report_path]:
            if os.path.exists(path):
                os.remove(path)
        await state.clear()
        await cmd_start(message, state)
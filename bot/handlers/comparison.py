from aiogram import Router, types, F
from aiogram.fsm.context import FSMContext
from aiogram.types import FSInputFile

from handlers.start import cmd_start
from keyboards.builders import KeyboardBuilder
from models.states import ComparisonStates
from services.analysis import analyze_documents
from services.file_processing import convert_to_pdf, convert_to_xls
from services.file_processing import process_pdf_file, cleanup_temp_files

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
    try:
        await callback.message.delete()
    except:
        pass

    category_name = {
        'arch_category': 'архитектурный раздел (АР)',
        'constr_category': 'конструктивный раздел (КР)',
        'eng_category': 'инженерный раздел (ИОС)',
        'water_supply_category': 'раздел водоснабжения (ИОС)',
        'water_drain_category': 'раздел водоотведения (ИОС)',
        'heat_network_category': 'раздел тепловых сетей и ИТП (ИОС)',
        'hvac_category': 'раздел ОВиК (ИОС)'
    }.get(callback.data, callback.data)

    await state.update_data(category=callback.data, category_name=category_name)
    await state.set_state(ComparisonStates.waiting_tz_file)
    await callback.message.answer("Пожалуйста, загрузите файл с ТЗ:",
                                  reply_markup=KeyboardBuilder.back_to_categories_kb())


@router.callback_query(F.data == 'back_to_categories')
async def back_to_categories(callback: types.CallbackQuery, state: FSMContext):
    user_data = await state.get_data()
    files_to_clean = [
        user_data.get('tz_file'),
        user_data.get('result_path'),
        user_data.get('report_path')
    ]

    await cleanup_temp_files([f for f in files_to_clean if f is not None])

    await state.set_state(ComparisonStates.choosing_category)
    await callback.message.edit_text(
        "Выберите раздел для сравнения:",
        reply_markup=KeyboardBuilder.categories_kb()
    )

@router.callback_query(ComparisonStates.waiting_format_choice, F.data.startswith('format_'))
async def handle_format_choice(callback: types.CallbackQuery, state: FSMContext):
    try:
        await callback.message.delete()
    except:
        pass
    report_path = pdf_path = xls_path = None
    user_data = await state.get_data()
    report_path = user_data.get('report_path')
    format_type = callback.data.split('_')[1]

    try:
        if format_type == 'txt':
            await callback.message.answer_document(
                FSInputFile(report_path),
                caption="Отчет в формате TXT"
            )
        elif format_type == 'pdf':
            pdf_path = await convert_to_pdf(report_path)
            await callback.message.answer_document(
                FSInputFile(pdf_path),
                caption="Отчет в формате PDF"
            )
        elif format_type == 'xls':
            xls_path = await convert_to_xls(report_path)
            await callback.message.answer_document(
                FSInputFile(xls_path),
                caption="Отчет в формате Excel"
            )

    except Exception as e:
        await callback.message.answer(f"Ошибка при подготовке отчета: {str(e)}")
    finally:
        files_to_clean = [
            user_data.get('tz_file'),
            user_data.get('result_path'),
            report_path,
            pdf_path,
            xls_path
        ]
        await cleanup_temp_files([f for f in files_to_clean if f is not None])
        await state.clear()
        await cmd_start(callback.message, state)



@router.message(
    ComparisonStates.waiting_tz_file,
    F.document & (F.document.mime_type == 'application/pdf')
)
async def handle_tz_file(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    file_path = await process_pdf_file(message, message.bot)
    await state.update_data(tz_file=file_path)

    category_name = user_data.get('category_name', 'выбранного раздела')

    await message.answer(f"Файл ТЗ успешно загружен. Теперь загрузите файл с результатом ({category_name}):",
                         reply_markup=KeyboardBuilder.back_to_categories_kb())
    await state.set_state(ComparisonStates.waiting_result_file)


@router.message(ComparisonStates.waiting_result_file)
async def handle_result_file(message: types.Message, state: FSMContext):
    result_path = None
    report_path = None
    user_data = await state.get_data()

    result_path = await process_pdf_file(message, message.bot)

    await state.update_data(result_path=result_path)

    category = user_data.get('category')

    await message.answer("Начинаю анализ... Ожидайте пожалуйста.")
    try:
        report_path = await analyze_documents(
            category=category,
            tz_path=user_data['tz_file'],
            result_path=result_path
        )

        await state.update_data(report_path=report_path)
        await state.set_state(ComparisonStates.waiting_format_choice)

        await message.answer(
            "Отчёт готов, выберите формат:",
            reply_markup=KeyboardBuilder.format_choice_kb()
        )


    except Exception as e:
        await message.answer(f"Произошла ошибка: {str(e)}")

        files_to_clean = [
            user_data.get('tz_file'),
            result_path,
            report_path
        ]

        await cleanup_temp_files([f for f in files_to_clean if f is not None])

        await state.clear()
        await cmd_start(message, state)
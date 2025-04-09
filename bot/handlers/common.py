from aiogram import Router, types, F
from aiogram.fsm.context import FSMContext
from bot.keyboards.builders import KeyboardBuilder

router = Router()

@router.callback_query(F.data == 'start')
async def back_to_start(callback: types.CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(
        "Главное меню:",
        reply_markup=KeyboardBuilder.start_kb()
    )
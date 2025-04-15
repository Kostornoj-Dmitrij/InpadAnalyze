from aiogram import Router, types, F
from aiogram.fsm.context import FSMContext
from bot.keyboards.builders import KeyboardBuilder

router = Router()

@router.callback_query(F.data == 'start')
async def back_to_start(callback: types.CallbackQuery, state: FSMContext):
    await state.clear()
    try:
        await callback.message.delete()
    except:
        pass
    await callback.message.answer(
        "Привет! Добро пожаловать в бота.", reply_markup=KeyboardBuilder.start_kb()
    )
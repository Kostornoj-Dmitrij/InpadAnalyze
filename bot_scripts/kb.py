from aiogram import types

# Клавиатура для старта
start_keyboard = types.InlineKeyboardMarkup(
    inline_keyboard=[
        [types.InlineKeyboardButton(text='Провести анализ документации', callback_data='analyze_docs')]
    ]
)

# Клавиатура для вопросов
question_keyboard = types.InlineKeyboardMarkup(
    inline_keyboard=[
        [types.InlineKeyboardButton(text='Задать вопрос ещё раз', callback_data='brusnika')],
        [types.InlineKeyboardButton(text='Назад', callback_data='start')]
    ]
)

# Клавиатура для выбора зданий
buildings_keyboard = types.InlineKeyboardMarkup(
    inline_keyboard=[
        [types.InlineKeyboardButton(text='Брусника А1', callback_data='brusnika')],
        [types.InlineKeyboardButton(text='Назад', callback_data='start')]
    ]
)

# Клавиатура для возврата в начало
back_to_start_keyboard = types.InlineKeyboardMarkup(
    inline_keyboard=[
        [types.InlineKeyboardButton(text='Назад', callback_data='start')]
    ]
)
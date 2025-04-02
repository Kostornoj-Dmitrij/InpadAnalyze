from aiogram import types

# Клавиатура для старта
start_keyboard = types.InlineKeyboardMarkup(
    inline_keyboard=[
        [types.InlineKeyboardButton(text='Провести анализ документации', callback_data='analyze_docs')],
        [types.InlineKeyboardButton(text='Задать вопрос по документу', callback_data='question')]
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

categories_keyboard = types.InlineKeyboardMarkup(
    inline_keyboard=[
        [types.InlineKeyboardButton(text='Архитектурный раздел', callback_data='arch_category')],
        [types.InlineKeyboardButton(text='Конструктивный раздел', callback_data='constr_category')],
        [types.InlineKeyboardButton(text='Инженерный раздел', callback_data='eng_category')],
        [types.InlineKeyboardButton(text='Назад', callback_data='start')]
    ]
)
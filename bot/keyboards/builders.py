from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardButton

class KeyboardBuilder:
    @staticmethod
    def start_kb():
        builder = InlineKeyboardBuilder()
        builder.add(
            InlineKeyboardButton(
                text='Провести анализ документации',
                callback_data='analyze_docs'
            ),
            InlineKeyboardButton(
                text='Задать вопрос по документу',
                callback_data='ask_question'
            )
        )
        builder.adjust(1)
        return builder.as_markup()

    @staticmethod
    def categories_kb():
        builder = InlineKeyboardBuilder()
        builder.add(
            InlineKeyboardButton(
                text='Архитектурный раздел',
                callback_data='arch_category'
            ),
            InlineKeyboardButton(
                text='Конструктивный раздел',
                callback_data='constr_category'
            ),
            InlineKeyboardButton(
                text='Инженерный раздел',
                callback_data='eng_options'
            ),
            InlineKeyboardButton(
                text='Назад',
                callback_data='start'
            )
        )
        builder.adjust(1)
        return builder.as_markup()

    @staticmethod
    def back_button_kb():
        builder = InlineKeyboardBuilder()
        builder.add(InlineKeyboardButton(text='Назад', callback_data='start'))
        return builder.as_markup()

    @staticmethod
    def eng_options_kb():
        builder = InlineKeyboardBuilder()
        builder.add(
            InlineKeyboardButton(
                text='Система водоснабжения',
                callback_data='water_supply_category'
            ),
            InlineKeyboardButton(
                text='Система водоотведения',
                callback_data='water_drain_category'
            ),
            InlineKeyboardButton(
                text='Тепловые сети и ИТП',
                callback_data='heat_network_category'
            ),
            InlineKeyboardButton(
                text='Отопление, вентиляция и кондиционирование',
                callback_data='hvac_category'
            ),
            InlineKeyboardButton(
                text='Без конкретного раздела (в общем)',
                callback_data='eng_category'
            ),
            InlineKeyboardButton(
                text='Назад',
                callback_data='back_to_categories'
            )
        )
        builder.adjust(1)
        return builder.as_markup()

    @staticmethod
    def format_choice_kb():
        builder = InlineKeyboardBuilder()
        builder.add(
            InlineKeyboardButton(text='📄 TXT', callback_data='format_txt'),
            InlineKeyboardButton(text='📝 PDF', callback_data='format_pdf'),
            InlineKeyboardButton(text='📊 Excel', callback_data='format_xls'),
        )
        builder.adjust(3)
        return builder.as_markup()

    @staticmethod
    def back_to_categories_kb():
        builder = InlineKeyboardBuilder()
        builder.add(
            InlineKeyboardButton(
                text='Назад к выбору раздела',
                callback_data='back_to_categories'
            )
        )
        return builder.as_markup()

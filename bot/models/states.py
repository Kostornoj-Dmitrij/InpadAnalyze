from aiogram.fsm.state import StatesGroup, State

class ComparisonStates(StatesGroup):
    choosing_category = State()
    waiting_tz_file = State()
    waiting_result_file = State()
    waiting_format_choice = State()

class QuestionStates(StatesGroup):
    waiting_file = State()
    waiting_question = State()
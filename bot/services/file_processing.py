import os
from aiogram.types import Message
from aiogram import Bot


async def process_pdf_file(message: Message, bot: Bot) -> str:
    if not os.path.exists("temp"):
        os.makedirs("temp")

    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = f"temp/{message.from_user.id}_{file.file_id}.pdf"

    await bot.download_file(file.file_path, destination=file_path)
    return file_path

async def cleanup_temp_files(file_paths: list):
    """Асинхронная очистка файлов"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"Удален временный файл: {path}")
            except Exception as e:
                print(f"Ошибка удаления файла {path}: {str(e)}")
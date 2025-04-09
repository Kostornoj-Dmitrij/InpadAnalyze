import asyncio
from PyPDF2 import PdfReader
from bot.config import OpenRouter_API_KEY
import aiohttp
import os


async def analyze_documents(tz_path: str, result_path: str, user_id: int) -> str:
    async def extract_text(path: str) -> str:
        with open(path, 'rb') as f:
            reader = PdfReader(f)
            return '\n'.join([page.extract_text() for page in reader.pages])

    tz_text, result_text = await asyncio.gather(
        extract_text(tz_path),
        extract_text(result_path)
    )

    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OpenRouter_API_KEY}"},
            json={
                "model": "deepseek/deepseek-chat:free",
                "messages": [{
                    "role": "system",
                    "content": "Сравни техническое задание и результат работы, найди все несоответствия."
                }, {
                    "role": "user",
                    "content": f"ТЗ:\n{tz_text}\n\nРезультат:\n{result_text}"
                }]
            }
        )
        result = await response.json()

    report_path = f"temp/{user_id}_report.txt"
    with open(report_path, 'w') as f:
        f.write(result['choices'][0]['message']['content'])

    return report_path


async def generate_answer(file_path: str, question: str) -> str:
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        text = '\n'.join([page.extract_text() for page in reader.pages])

    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OpenRouter_API_KEY}"},
            json={
                "model": "deepseek/deepseek-chat:free",
                "messages": [{
                    "role": "system",
                    "content": "Ответь на вопрос пользователя на основе предоставленного документа."
                }, {
                    "role": "user",
                    "content": f"Документ:\n{text}\n\nВопрос: {question}"
                }]
            }
        )
        result = await response.json()

    return result['choices'][0]['message']['content']
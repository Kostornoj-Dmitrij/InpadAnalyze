from typing import List, Coroutine, Awaitable, TypedDict, NamedTuple

import aiofiles
from aiogram.client.session import aiohttp
from tqdm.asyncio import tqdm_asyncio
from aiogram import types
from main import bot, user_data
import kb
import os
import asyncio
from datetime import datetime
import requests
from PyPDF2 import PdfReader
from config import OpenRouter_API_KEY, OPENAI_API_KEY
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
import tiktoken
from concurrent.futures import ThreadPoolExecutor

class Page(NamedTuple):
    num: int
    text: str

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
executor = ThreadPoolExecutor(max_workers=4)

semaphore = asyncio.Semaphore(3)

async def show_buildings_options(user_id):
    keyboard = kb.buildings_keyboard
    await bot.send_message(user_id, "Выберите здание, по которому хотите задать вопрос:", reply_markup=keyboard)

async def user_clear(message):
    user_data[message.chat.id].state    = 'chat_start'
    user_data[message.chat.id].revit_choice = 'chat_start'
    user_data[message.chat.id].feedback_text = ''
    user_data[message.chat.id].license_key = ''
    user_data[message.chat.id].build_version = ''
    user_data[message.chat.id].revit_version = ''
    user_data[message.chat.id].choice = 'chat_start'
    user_data[message.chat.id].file_path = ''
    user_data[message.chat.id].photo_path = ''
    user_data[message.chat.id].renga_version = ''
    user_data[message.chat.id].plugin_id = ''
    user_data[message.chat.id].plugin_category = 'chat_start'
    user_data[message.chat.id].plugins_build = ''


async def send_long_message(chat_id, text, chunk_size=4096):
    for i in range(0, len(text), chunk_size):
        await bot.send_message(chat_id, text[i:i + chunk_size])


def count_tokens(text):
    """Подсчитывает количество токенов в тексте."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

async def extract_text_from_pdf(pdf_path: str) -> List[Page]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, sync_extract_text, pdf_path)

def sync_extract_text(pdf_path):
    text = []
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            text.append(Page(num=page_num, text=f"\n--- Страница {page_num} ---\n{page_text}"))
    return text


async def split_text(text, max_tokens=30000, overlap=200):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, sync_split_text, text, max_tokens, overlap)


def sync_split_text(pages: List[Page], max_tokens=30000, overlap=200) -> List[str]:
    """Разделяет текст на чанки с сохранением информации о страницах."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_pages = set()

    for page in pages:
        page_tokens = len(tokenizer.encode(page.text))

        # Если добавление текущей страницы превысит лимит
        if current_tokens + page_tokens > max_tokens:
            if current_chunk:
                # Сохраняем текущий чанк
                chunk_text = "\n".join([p.text for p in current_chunk])
                page_nums = sorted(current_pages)
                header = f"Страницы: {', '.join(map(str, page_nums))}\n"
                chunks.append(header + chunk_text)

                # Создаем перекрытие
                if overlap > 0:
                    overlap_pages = current_chunk[-overlap:]
                    current_chunk = overlap_pages
                    current_tokens = sum(len(tokenizer.encode(p.text)) for p in overlap_pages)
                    current_pages = {p.num for p in overlap_pages}
                else:
                    current_chunk = []
                    current_tokens = 0
                    current_pages = set()
            else:
                # Страница слишком большая для одного чанка - разбиваем
                page_lines = page.text.split('\n')
                line_tokens = [len(tokenizer.encode(line)) for line in page_lines]

                temp_chunk = []
                temp_tokens = 0
                for line, tokens in zip(page_lines, line_tokens):
                    if temp_tokens + tokens > max_tokens:
                        if temp_chunk:
                            chunks.append("\n".join(temp_chunk))
                            temp_chunk = []
                            temp_tokens = 0
                    temp_chunk.append(line)
                    temp_tokens += tokens

                if temp_chunk:
                    chunks.append("\n".join(temp_chunk))
                continue

        current_chunk.append(page)
        current_pages.add(page.num)
        current_tokens += page_tokens

    # Добавляем последний чанк
    if current_chunk:
        chunk_text = "\n".join([p.text for p in current_chunk])
        page_nums = sorted(current_pages)
        header = f"Страницы: {', '.join(map(str, page_nums))}\n"
        chunks.append(header + chunk_text)

    return chunks


async def create_vector_db(text_chunks):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_db = None

    def add_to_db(db, chunk):
        if db is None:
            return FAISS.from_texts([chunk], embeddings)
        db.add_texts([chunk])
        return db

    for chunk in text_chunks:
        if not chunk.strip():
            continue

        async with semaphore:
            try:
                vector_db = await asyncio.to_thread(
                    add_to_db,
                    vector_db,
                    chunk
                )
            except Exception as e:
                print(f"Ошибка при обработке чанка: {e}")
                continue

        await asyncio.sleep(1)

    return vector_db

async def find_similar_sections(query, vector_db, k=2):
    """Находит похожие разделы в векторной базе данных."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, vector_db.similarity_search, query, k)

async def compare_documents(technical_spec, result_doc):
    """Сравнивает разделы ТЗ и результата."""
    async with aiohttp.ClientSession() as session:
        messages = [
            {"role": "system",
             "content": "Ты эксперт по анализу документов. Твоя задача - сравнивать техническое задание с результатом работы и находить ВСЕ несоответствия."},
            {"role": "user", "content": f"Вот часть технического задания:\n{technical_spec}\n\n"
                                      f"Вот соответствующий результат работы:\n{result_doc}\n\n"
                                      "Требуется:\n"
                                      "1. Найти ВСЕ несоответствия между ТЗ и результатом\n"
                                      "2. Для каждого несоответствия УКАЗЫВАТЬ ТОЧНЫЕ НОМЕРА СТРАНИЦ из обоих документов\n"
                                      "3. Формат вывода для каждого пункта:\n"
                                      "   - [ТЗ стр.X] [Результат стр.Y] Описание несоответствия\n"
                                      "   - Нормативный документ (если применимо)\n"
                                      "   - Рекомендации по исправлению\n\n"
                                      "4. Отдельно указать, что выполнено правильно\n"
                                      "5. Особое внимание уделять:\n"
                                      "   - Нормативным требованиям\n"
                                      "   - Техническим параметрам\n"
                                      "   - Конструктивным решениям\n"
                                      "   - Материалам и их характеристикам"}
        ]
        async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OpenRouter_API_KEY}", "Content-Type": "application/json"},
                json={"model": "deepseek/deepseek-chat:free", "messages": messages}, timeout=600
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                return ""


async def process_chunk(result_chunk, technical_db):
    """Обрабатывает чанк результата и сравнивает с похожими разделами ТЗ."""
    # Извлекаем номера страниц из заголовка чанка
    result_pages = set()
    first_line = result_chunk.split('\n')[0]
    if first_line.startswith('Страницы:'):
        page_nums = first_line.replace('Страницы:', '').strip()
        result_pages.update(map(int, page_nums.split(',')))

    similar_sections = await find_similar_sections(result_chunk, technical_db)
    tasks = [compare_documents(section, result_chunk) for section in similar_sections]
    results = await asyncio.gather(*tasks)

    # Добавляем информацию о страницах результата к каждому ответу
    if result_pages:
        page_info = f"\n(Анализируемые страницы результата: {', '.join(map(str, sorted(result_pages)))})"
        results = [r + page_info for r in results]

    return results


async def split_text_summarized(text, max_tokens=10000) -> List[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        sync_split_text_summarized,
        text,
        max_tokens
    )

def sync_split_text_summarized(text, max_tokens=10000) -> List[str]:
    """Разделяет текст на части по количеству токенов."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))
        start = end
    return chunks

async def structure_report_part(report_part):
    """Структурирует часть отчёта."""
    async with aiohttp.ClientSession() as session:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты эксперт по обработке и структурированию текстов. "
                    "Ты получаешь часть готового отчёта, который уже содержит анализ технического задания и выполненной работы. "
                    "Отчёт состоит из нескольких частей, и твоя задача — обработать текущую часть, не теряя контекста. "
                    "Важно: ты должен включить ВСЕ несоответствия и проблемы из этой части. "
                    "Не удаляй и не изменяй информацию, только структурируй её."
                )
            },
            {
                "role": "user",
                "content": (
                    "Вот часть отчёта по анализу документов:\n{report_part}\n\n"
                    "Структурируй её, выделив следующие разделы:\n"
                    "1. **Ключевые несоответствия** — перечисли ВСЕ несоответствия между техническим заданием и выполненной работой. "
                    "Не удаляй и не изменяй формулировки из исходного отчёта.\n"
                    "2. **Рекомендации** — перечисли рекомендации по устранению несоответствий.\n"
                    "3. **Технические ошибки** — перечисли технические ошибки, если они есть.\n"
                    "4. **Нарушения нормативных требований** — перечисли нарушения, если они есть.\n"
                    "5. **Положительные моменты** — кратко упомяни, что выполнено верно.\n"
                    "Сделай текст более читаемым, используй маркированные списки и чёткие формулировки. "
                    "Не добавляй новую информацию И НЕ УДАЛЯЙ! НОМЕРА СТРАНИЦЫ ТОЖЕ ДОЛЖНЫ БЫТЬ ОСТАВЛЕНЫ."
                ).format(report_part=report_part)
            }
        ]
        async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OpenRouter_API_KEY}"},
                json={"model": "deepseek/deepseek-chat:free", "messages": messages},
                timeout=600
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data["choices"][0]["message"]["content"]
            return ""

async def final_structure_report(structured_parts):
    """Финальная структуризация объединённого отчёта."""
    combined_report = "\n\n".join(structured_parts)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты эксперт по обработке и структурированию текстов. "
                "Ты получаешь объединённый отчёт, который состоит из нескольких частей. "
                "Твоя задача — удалить дубликаты, объединить похожие пункты и сделать отчёт более читаемым. "
                "Не удаляй и не изменяй информацию, только улучшай её структуру."
            )
        },
        {
            "role": "user",
            "content": (
                "Вот объединённый отчёт:\n{combined_report}\n\n"
                "Структурируй его, выделив следующие разделы:\n"
                "1. **Ключевые несоответствия** — объедини похожие пункты и удали дубликаты.\n"
                "2. **Рекомендации** — объедини похожие пункты и удали дубликаты.\n"
                "3. **Технические ошибки** — объедини похожие пункты и удали дубликаты.\n"
                "4. **Нарушения нормативных требований** — объедини похожие пункты и удали дубликаты.\n"
                "5. **Положительные моменты** — объедини похожие пункты и удали дубликаты.\n"
                "Сделай текст более читаемым, используй маркированные списки и чёткие формулировки. "
                "Не добавляй новую информацию И НЕ УДАЛЯЙ! НОМЕРА СТРАНИЦЫ ТОЖЕ ДОЛЖНЫ БЫТЬ ОСТАВЛЕНЫ."
            ).format(combined_report=combined_report)
        }
    ]
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OpenRouter_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "deepseek/deepseek-chat:free",
                        "messages": messages
                    },
                    timeout=600
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                return f"Ошибка API: {await response.text()}"
        except Exception as e:
            return f"Ошибка соединения: {str(e)}"


async def analyze_files(user_id):
    user = user_data[user_id]
    initial_report_path = ""
    final_report_path = ""
    await bot.send_message(user_id, "Начинаю анализ документов... Это может занять некоторое время.")

    empty_report_path = f"temp/{user_id}_empty_report.txt"

    async with aiofiles.open(empty_report_path, "w", encoding="utf-8") as f:
        await f.write("Отчёт")  # Записываем пустую строку

    async with aiofiles.open(empty_report_path, "rb") as report_file:
        await bot.send_document(
            user_id,
            types.FSInputFile(empty_report_path),
            caption="✅ Отчёт готов"
        )
    return
    try:
        # Извлекаем текст из PDF
        technical_text, result_text = await asyncio.gather(
            extract_text_from_pdf(user.tz_file_path),
            extract_text_from_pdf(user.result_file_path)
        )
        print("Текст извлечён")

        technical_chunks, result_chunks = await asyncio.gather(
            split_text(technical_text),
            split_text(result_text)
        )
        print("Разделен на чанки")
        technical_db = await create_vector_db(technical_chunks)
        print("Создана векторная база")
        all_results = []
        progress = tqdm_asyncio(result_chunks, desc="Анализ чанков")
        print("Анализ чанков")
        try:
            async for chunk in progress:
                results = await process_chunk(chunk, technical_db)
                all_results.extend(results)
                await asyncio.sleep(0)  # Даем обработать другие задачи
        except Exception as e:
            await bot.send_message(user_id, f"❌ Ошибка анализа: {type(e).__name__}: {str(e)}")
            raise
        print("Чанки проанализированы")
        initial_report = "\n\n".join(all_results)

        initial_report_path = f"temp/{user_id}_initial_report.txt"

        async with aiofiles.open(initial_report_path, "w") as f:
            await f.write(initial_report)
        print("Сохранён отчёт")
        async with aiofiles.open(initial_report_path, "r", encoding="utf-8") as f:
            report_content = await f.read()

        report_parts: List[str] = await split_text_summarized(report_content)
        print("Обработка частей отчёта")
        # Параллельная обработка частей отчёта
        structured_tasks: List[Awaitable[str]] = [
            structure_report_part(part)
            for part in report_parts  # Теперь итерируемся по реальному списку
        ]
        structured_parts: List[str] = await asyncio.gather(*structured_tasks)
        print("Финальная структуризация")
        # Финальная структуризация
        final_report = await final_structure_report(structured_parts)
        print("Сохранение файлов")
        # Асинхронное сохранение и отправка
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        final_report_path = f"temp/{user_id}_final_{current_time}.txt"

        async with aiofiles.open(final_report_path, "w", encoding="utf-8") as f:
            await f.write(final_report)

        async with aiofiles.open(final_report_path, "rb") as report_file:
            content = await report_file.read()
            await bot.send_document(
                user_id,
                types.FSInputFile(final_report_path),  # Используем готовый файл
                caption="✅ Финальный отчёт готов"
            )

    except Exception as e:
        print(e)
        await bot.send_message(user_id, f"Произошла ошибка при анализе документов: {str(e)}")
    finally:
        cleanup_tasks = []
        paths_to_clean = []

        if hasattr(user, 'tz_file_path') and user.tz_file_path:
            paths_to_clean.append(user.tz_file_path)
        if hasattr(user, 'result_file_path') and user.result_file_path:
            paths_to_clean.append(user.result_file_path)
        if initial_report_path and os.path.exists(initial_report_path):
            paths_to_clean.append(initial_report_path)
        if final_report_path and os.path.exists(final_report_path):
            paths_to_clean.append(final_report_path)

        for path in paths_to_clean:
            try:
                cleanup_tasks.append(asyncio.to_thread(os.remove, path))
            except Exception as e:
                print(f"Ошибка при удалении {path}: {e}")

        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
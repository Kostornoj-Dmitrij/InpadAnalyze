import os
import asyncio
from datetime import datetime

import requests
from PyPDF2 import PdfReader
from bot.config import OpenRouter_API_KEY, OPENAI_API_KEY
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
import tiktoken


def extract_sections_from_ts(text: str) -> dict:
    """
    Разбивает текст ТЗ на логические разделы по заголовкам.
    Возвращает словарь вида {"Архитектурные решения": "...", "Конструктивные решения": "...", ...}
    """
    import re
    section_titles = [
        "Архитектурно-строительные решения",
        "Конструктивные решения",
        "Инженерные системы",
        "Отделка помещений",
        "Организация строительства",
        "Мероприятия по охране окружающей среды",
        "Энергетическая эффективность",
    ]

    sections = {}
    pattern = '|'.join(re.escape(title) for title in section_titles)

    # Найдём все заголовки разделов
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group().strip()
        content = text[start:end].strip()
        sections[title] = content

    return sections




def count_tokens(text):
    """Подсчитывает количество токенов в тексте."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Извлекает текст с явными метками страниц"""
    reader = PdfReader(pdf_path)
    text = []
    for page_num, page in enumerate(reader.pages, start=1):
        content = page.extract_text() or ""
        text.append(f"\n=== НАЧАЛО СТРАНИЦЫ {page_num} ===\n{content}\n=== КОНЕЦ СТРАНИЦЫ {page_num} ===")
    return "\n".join(text)

def split_text(text, max_tokens=28000, overlap=1000):
    """Разделяет текст на чанки с перекрытием, сохраняя информацию о страницах."""
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Разделяем текст на страницы
    pages = []
    current_page = None
    for line in text.split('\n'):
        if line.startswith('=== НАЧАЛО СТРАНИЦЫ'):
            if current_page is not None:
                pages.append(current_page)
            current_page = {'num': int(line.split()[3]), 'text': ''}
        elif current_page is not None and not line.startswith('=== КОНЕЦ СТРАНИЦЫ'):
            current_page['text'] += line + '\n'
    if (current_page is not

            None):
        pages.append(current_page)

    # Собираем чанки с информацией о страницах
    chunks = []
    current_chunk = ""
    current_chunk_pages = set()
    current_tokens = 0

    for page in pages:
        page_text = f"\n\n=== СТРАНИЦА {page['num']} ===\n{page['text']}"
        page_tokens = len(tokenizer.encode(page_text))

        if current_tokens + page_tokens > max_tokens:
            # Добавляем текущий чанк
            if current_chunk:
                chunk_header = f"Страницы: {', '.join(map(str, sorted(current_chunk_pages)))}\n"
                chunks.append(chunk_header + current_chunk)

            # Начинаем новый чанк с перекрытием
            current_chunk = ""
            current_chunk_pages = set()
            current_tokens = 0

            # Добавляем часть предыдущей страницы для перекрытия
            if overlap > 0 and chunks:
                prev_text = chunks[-1]
                overlap_text = tokenizer.decode(tokenizer.encode(prev_text)[-overlap:])
                current_chunk = overlap_text + "\n"
                current_tokens = overlap

        current_chunk += page_text
        current_chunk_pages.add(page['num'])
        current_tokens += page_tokens

    # Добавляем последний чанк
    if current_chunk:
        chunk_header = f"Страницы: {', '.join(map(str, sorted(current_chunk_pages)))}\n"
        chunks.append(chunk_header + current_chunk)

    return [chunk for chunk in chunks if chunk.strip()]

def create_vector_db(text_chunks):
    """Создаёт векторную базу данных для семантического поиска."""
    if not text_chunks:
        raise ValueError("Не получено данных для обработки")

        # Фильтруем пустые чанки
    valid_chunks = [chunk for chunk in text_chunks if chunk.strip()]

    if not valid_chunks:
        raise ValueError("Все чанки пусты. Невозможно создать векторную базу")

    try:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        return FAISS.from_texts(valid_chunks, embeddings)
    except Exception as e:
        print(f"Ошибка создания векторной базы: {e}")
        raise

def find_similar_sections(query, vector_db, k=2):
    """Находит похожие разделы в векторной базе данных."""
    return vector_db.similarity_search(query, k=k)

async def compare_documents(technical_spec, result_doc):
    """Сравнивает разделы ТЗ и результата."""
    messages = [
        {"role": "system",
         "content": "Ты эксперт по анализу документов. При анализе обязательно указывай номера страниц, на которых найдены несоответствия."},
        {"role": "user", "content": f"Вот часть технического задания:\n{technical_spec}\n\n"
                                    f"Вот соответствующий результат работы:\n{result_doc}\n\n"
                                    "Найди несоответствия и укажи, что выполнено правильно, а что — нет. "
                                    "Обязательно указывай номера страниц, на которых найдены проблемы (И ИЗ ТЗ И ИЗ РЕЗУЛЬТАТА РАБОТЫ, ЧТОБЫ МОЖНО БЫЛО СРАВНИТЬ)"
                                    "Особое внимание удели всем материалам в стенах и полах, а также толщине стен"
                                    }
    ]
    try:
        print("Отправлен запрос")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OpenRouter_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek/deepseek-chat:free", "messages": messages}, timeout=30
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Ошибка при запросе к API: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        print(f"Ошибка при сравнении документов: {e}")
        return ""


async def process_chunk(result_chunk, technical_db):
    """Обрабатывает чанк результата и сравнивает с похожими разделами ТЗ."""
    # Извлекаем номера страниц из чанка результата
    result_pages = set()
    for line in result_chunk.split('\n'):
        if line.startswith('=== СТРАНИЦА'):
            result_pages.add(int(line.split()[2]))
        elif line.startswith('=== НАЧАЛО СТРАНИЦЫ'):
            result_pages.add(int(line.split()[2]))

    similar_sections = find_similar_sections(result_chunk, technical_db)
    tasks = [compare_documents(section, result_chunk) for section in similar_sections]
    results = await asyncio.gather(*tasks)

    # Добавляем информацию о страницах результата к каждому ответу
    if result_pages:
        pages_sorted = sorted(result_pages)
        if len(pages_sorted) > 1:
            page_info = f"\n(Анализируемые страницы результата: {pages_sorted[0]}-{pages_sorted[-1]})"
        else:
            page_info = f"\n(Анализируемая страница результата: {pages_sorted[0]})"

        results = [r + page_info for r in results if not r.strip().endswith(')')]

    return results

def split_text_summarized(text, max_tokens=13000):
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
                "2. **Технические ошибки** — перечисли технические ошибки, если они есть.\n"
                "3. **Нарушения нормативных требований** — перечисли нарушения, если они есть.\n"
                "НЕ УДАЛЯЙ и не изменяй формулировки из исходного отчёта.\n"
                "КАЖДОЕ НЕСООТВЕТСТВИЕ ДОЛЖНО ССЫЛАТЬСЯ НА СТРАНИЦУ И ИЗ ТЗ И ИЗ РЕЗУЛЬТА РАБОТЫ\n"

                
                "Сделай текст более читаемым, используй маркированные списки и чёткие формулировки. "
                "Не добавляй новую информацию."
            ).format(report_part=report_part)
        }
    ]
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OpenRouter_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek/deepseek-chat:free", "messages": messages},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Ошибка при запросе к API: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        print(f"Ошибка при структурировании части отчёта: {e}")
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
                "1. **Ключевые несоответствия** — перечисли ВСЕ несоответствия между техническим заданием и выполненной работой. "
                "2. **Технические ошибки** — перечисли технические ошибки, если они есть.\n"
                "3. **Нарушения нормативных требований** — перечисли нарушения, если они есть.\n"
                "НЕ УДАЛЯЙ и не изменяй формулировки из исходного отчёта.\n"
                "КАЖДОЕ НЕСООТВЕТСТВИЕ ДОЛЖНО ССЫЛАТЬСЯ НА СТРАНИЦУ И ИЗ ТЗ И ИЗ РЕЗУЛЬТА РАБОТЫ\n"
                "Сделай текст более читаемым, используй маркированные списки и чёткие формулировки. "
                "Не добавляй новую информацию."
            ).format(combined_report=combined_report)
        }
    ]
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OpenRouter_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek/deepseek-chat:free", "messages": messages},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Ошибка при запросе к API: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        print(f"Ошибка при финальной структуризации отчёта: {e}")
        return ""

async def main():
    # Загрузка текстов из PDF
    print('Загрузка текстов')
    technical_text = extract_text_from_pdf("data/new_data/ТЗ.pdf")
    result_text = extract_text_from_pdf("data/new_data/АР.pdf")
    technical_tokens = count_tokens(technical_text)
    result_tokens = count_tokens(result_text)
    print(f"Токенов в техническом задании: {technical_tokens}")
    print(f"Токенов в результате: {result_tokens}")
    # Разбиение текстов на части

    ts_sections = extract_sections_from_ts(technical_text)
    section_name = "Архитектурно-строительные решения"  # <-- сюда можно подавать через аргумент
    technical_section_text = ts_sections.get(section_name)
    print("Первые 100 символов раздела:\n", technical_section_text[:100])
    print("Последние 100 символов раздела:\n", technical_section_text[-100:])

    print('Разбиение текстов на части')
    technical_chunks = split_text(technical_section_text)
    result_chunks = split_text(result_text)
    print('Создание векторных хранилищ')
    # Создание векторных хранилищ
    technical_db = create_vector_db(technical_chunks)
    print('Сбор всех результатов сравнения')
    # Сбор всех результатов сравнения
    all_results = []

    # Прогресс-бар для отслеживания выполнения
    with tqdm(total=len(result_chunks), desc="Сравнение документов") as pbar:
        tasks = [process_chunk(chunk, technical_db) for chunk in result_chunks]
        for task in asyncio.as_completed(tasks):  # Обработка чанков параллельно
            results = await task
            all_results.extend(results)
            pbar.update(1)
            await asyncio.sleep(1)

    # Формирование общего отчета\
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_report = "\n\n".join(all_results)
    print("\nИтоговый отчет по анализу и сравнению:\n")
    print(final_report)


    # Сохранение отчета в файл (опционально)
    with open(f"deepseek_results/final_report_deepseek_{current_time}.txt", "w", encoding="utf-8") as file:
        file.write(final_report)
    with open(f"deepseek_results/final_report_deepseek_{current_time}.txt", "r", encoding="utf-8") as file:
        report = file.read()

    report_parts = split_text_summarized(report)
    print(f"Отчёт разделён на {len(report_parts)} частей.")

    structured_parts = []
    for i, part in enumerate(report_parts):
        print(f"Структурирование части {i + 1}...")
        structured_part = await structure_report_part(part)
        structured_parts.append(structured_part)

    # Финальная структуризация объединённого отчёта
    print("Финальная структуризация отчёта...")
    final_report = await final_structure_report(structured_parts)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_filename = f"deepseek_results/final_structured_report_deepseek_{current_time}.txt"
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(final_report)
    print(f"Отчёт сохранён в файл: {output_filename}")

# Запуск асинхронного main
asyncio.run(main())
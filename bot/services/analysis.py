import asyncio
import re
from datetime import datetime

import aiofiles
import aiohttp
import tiktoken
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from tqdm.asyncio import tqdm_asyncio
from bot.services.key_manager import key_manager
from bot.config import OPENAI_API_KEY


async def extract_requirements(text: str, mode: str) -> str:
    """Извлекает инженерные требования по частям"""
    chunks = await split_text(text, max_tokens=28000, overlap=500)
    all_requirements = []

    semaphore = asyncio.Semaphore(3)
    async def process_chunk(chunk: str):
        nonlocal mode
        async with semaphore:
            if mode == 'arch':
                messages = [
                    {
                        "role": "system",
                        "content": "Извлекай ТОЛЬКО информацию архитектурно-строительного раздела (материалы стен и полов, перегородки, фасады и т. д.) с указанием страниц."
                    },
                    {
                        "role": "user",
                        "content": f"Извлеки архитектурно-строительные требования (если они есть) из этого фрагмента:\n{chunk}"
                                    "Если в тексте есть отметки страниц: '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' ИЛИ '=== КОНЕЦ СТРАНИЦЫ {page_num} ===', и ты нашёл на этой странице какое-то требование, то НИ В КОЕМ СЛУЧАЕ НЕ УДАЛЯЙ МЕТКИ, мне нужно чтобы найденные тобой требования были абсолютно в том же виде окантованы этими метками! ВСЕ МЕТКИ СТРАНИЦ ДОЛЖНЫ БЫТЬ В ФОРМАТЕ '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' текст страницы... '=== КОНЕЦ СТРАНИЦЫ {page_num} ===')"                    }
                ]
            elif mode == 'structural':
                messages = [
                    {
                        "role": "system",
                        "content": "Извлекай ТОЛЬКО информацию конструктивного раздела (материалы, толщины, армирование и т. д.) с указанием страниц."
                    },
                    {
                        "role": "user",
                        "content": f"Извлеки конструктивные требования (если они есть) из этого фрагмента:\n{chunk}"
                                   "Если в тексте есть отметки страниц: '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' ИЛИ '=== КОНЕЦ СТРАНИЦЫ {page_num} ===', и ты нашёл на этой странице какое-то требование, то НИ В КОЕМ СЛУЧАЕ НЕ УДАЛЯЙ МЕТКИ, мне нужно чтобы найденные тобой требования были абсолютно в том же виде окантованы этими метками! ВСЕ МЕТКИ СТРАНИЦ ДОЛЖНЫ БЫТЬ В ФОРМАТЕ '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' текст страницы... '=== КОНЕЦ СТРАНИЦЫ {page_num} ===')"
                    }
                ]
            elif mode == 'water_supply':
                messages = [
                    {
                        "role": "system",
                        "content": "Извлекай ТОЛЬКО требования к системе водоснабжения (трубопроводы, насосы, фильтры, производители оборудования и т. д.) с указанием страниц."
                    },
                    {
                        "role": "user",
                        "content": f"Извлеки требования к системе водоснабжения(если они есть) из этого фрагмента:\n{chunk}"
                                   "Если в тексте есть отметки страниц: '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' ИЛИ '=== КОНЕЦ СТРАНИЦЫ {page_num} ===', и ты нашёл на этой странице какое-то требование, то НИ В КОЕМ СЛУЧАЕ НЕ УДАЛЯЙ МЕТКИ, мне нужно чтобы найденные тобой требования были абсолютно в том же виде окантованы этими метками! ВСЕ МЕТКИ СТРАНИЦ ДОЛЖНЫ БЫТЬ В ФОРМАТЕ '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' текст страницы... '=== КОНЕЦ СТРАНИЦЫ {page_num} ===')"
                    }
                ]
            elif mode == 'water_drain':
                messages = [
                    {
                        "role": "system",
                        "content": "Извлекай ТОЛЬКО требования к системе водоотведения (канализация, очистные сооружения, материалы труб, производители оборудования и т. д.) с указанием страниц."
                    },
                    {
                        "role": "user",
                        "content": f"Извлеки требования к системе водоотведения(если они есть) из этого фрагмента:\n{chunk}"
                                   "Если в тексте есть отметки страниц: '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' ИЛИ '=== КОНЕЦ СТРАНИЦЫ {page_num} ===', и ты нашёл на этой странице какое-то требование, то НИ В КОЕМ СЛУЧАЕ НЕ УДАЛЯЙ МЕТКИ, мне нужно чтобы найденные тобой требования были абсолютно в том же виде окантованы этими метками! ВСЕ МЕТКИ СТРАНИЦ ДОЛЖНЫ БЫТЬ В ФОРМАТЕ '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' текст страницы... '=== КОНЕЦ СТРАНИЦЫ {page_num} ===')"
                    }
                ]
            elif mode == 'heat_network':
                messages = [
                    {
                        "role": "system",
                        "content": "Извлекай ТОЛЬКО требования к тепловым сетям и ИТП (температурные режимы, производители оборудования, схемы подключений и т. д.) с указанием страниц."
                    },
                    {
                        "role": "user",
                        "content": f"Извлеки требования к тепловым сетям и ИТП(если они есть) из этого фрагмента:\n{chunk}"
                                   "Если в тексте есть отметки страниц: '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' ИЛИ '=== КОНЕЦ СТРАНИЦЫ {page_num} ===', и ты нашёл на этой странице какое-то требование, то НИ В КОЕМ СЛУЧАЕ НЕ УДАЛЯЙ МЕТКИ, мне нужно чтобы найденные тобой требования были абсолютно в том же виде окантованы этими метками! ВСЕ МЕТКИ СТРАНИЦ ДОЛЖНЫ БЫТЬ В ФОРМАТЕ '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' текст страницы... '=== КОНЕЦ СТРАНИЦЫ {page_num} ===')"
                    }
                ]
            elif mode == 'hvac':
                messages = [
                    {
                        "role": "system",
                        "content": "Извлекай ТОЛЬКО требования к отоплению, вентиляции и кондиционированию(типы систем, производители оборудования, параметры воздуха и т. д.) с указанием страниц."
                    },
                    {
                        "role": "user",
                        "content": f"Извлеки требования к системам отопления, вентиляции и кондиционирования(если они есть) из этого фрагмента:\n{chunk}"
                                   "Если в тексте есть отметки страниц: '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' ИЛИ '=== КОНЕЦ СТРАНИЦЫ {page_num} ===', и ты нашёл на этой странице какое-то требование, то НИ В КОЕМ СЛУЧАЕ НЕ УДАЛЯЙ МЕТКИ, мне нужно чтобы найденные тобой требования были абсолютно в том же виде окантованы этими метками! ВСЕ МЕТКИ СТРАНИЦ ДОЛЖНЫ БЫТЬ В ФОРМАТЕ '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' текст страницы... '=== КОНЕЦ СТРАНИЦЫ {page_num} ===')"
                    }
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": "Извлекай ТОЛЬКО информацию инженерного раздела (производители оборудования отопления, вентиляции, кондиционирования, водоснабжения и т. д.) с указанием страниц."
                    },
                    {
                        "role": "user",
                        "content": f"Извлеки инженерные требования (если они есть) из этого фрагмента:\n{chunk}"
                                    "Если в тексте есть отметки страниц: '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' ИЛИ '=== КОНЕЦ СТРАНИЦЫ {page_num} ===', и ты нашёл на этой странице какое-то требование, то НИ В КОЕМ СЛУЧАЕ НЕ УДАЛЯЙ МЕТКИ, мне нужно чтобы найденные тобой требования были абсолютно в том же виде окантованы этими метками! ВСЕ МЕТКИ СТРАНИЦ ДОЛЖНЫ БЫТЬ В ФОРМАТЕ '=== НАЧАЛО СТРАНИЦЫ {page_num} ===' текст страницы... '=== КОНЕЦ СТРАНИЦЫ {page_num} ===')"                    }
                ]
            retries = 3
            while retries > 0:
                current_key = await key_manager.get_key()
                if not current_key:
                    await asyncio.sleep(1)
                    print("Все ключи исчерпаны")
                    return None
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {current_key}",
                                     "Content-Type": "application/json"},
                            json={"model": "deepseek/deepseek-chat:free", "messages": messages},
                            timeout=aiohttp.ClientTimeout(total=150)
                        ) as response:
                            await key_manager.update_usage(current_key, response)

                            if response.status == 200:
                                result = await response.json()
                                if 'choices' in result and len(result['choices']) > 0:
                                    return result['choices'][0]['message']['content'].strip()
                            elif response.status == 429:
                                await asyncio.sleep(1)
                                print(f"Ключ {current_key[-5:]}... исчерпан")
                                await key_manager.handle_error(current_key)
                                retries -= 1
                                continue
                            else:
                                await asyncio.sleep(1)
                                print(f"Ошибка {response.status} с ключом {current_key[-5:]}...")
                                return None
                except Exception as e:
                    await asyncio.sleep(1)
                    print(f"Ошибка с ключом {current_key[-5:]}...: {str(e)}")
                    await key_manager.handle_error(current_key)
                    retries -= 1

    tasks =[process_chunk(chunk) for chunk in chunks]
    for future in tqdm_asyncio.as_completed(tasks, desc="Извлечение требований"):
        result = await future
        if result:
            all_requirements.append(result)
        await asyncio.sleep(0.5)

    return "\n".join(filter(None, all_requirements))

async def extract_sections_from_ts(text: str, mode: str = "default") -> dict:
    """Разбивает текст ТЗ на разделы с возможностью нейросетевого анализа"""
    sections = {}

    ALL_SECTION_TITLES = [
        "Основные данные",
        "Архитектурно-строительные решения",
        "Отделка помещений",
        "Конструктивные решения",
        "Инженерные системы",
        "Инженерные решения",
        "Раздел «Отопление, вентиляция и кондиционирование»",
        "Раздел «Водоснабжение и канализация»",
        "Раздел «Слаботочные системы»",
        "Схема планировочной организации земельного участка",
    ]

    TARGET_TITLES = {
        "arch": ["Архитектурно-строительные решения"],
        "structural": ["Конструктивные решения"],
        "engineer": ["Инженерные системы"],
        "water_supply": ["Водоснабжение и канализация"],
        "water_drain": ["Водоснабжение и канализация"],
        "heat_network": ["Отопление, вентиляция и кондиционирование"],
        "hvac": ["Отопление, вентиляция и кондиционирование"]
    }

    if mode == "engineer":
        print("Выбран нейросетевой парсинг для инженерного раздела")
        structural_text = await extract_requirements(text, mode)
        if structural_text:
            sections["Инженерные системы"] = structural_text
        print("Требования извлечены")
        return sections


    all_pattern = '|'.join(re.escape(title) for title in ALL_SECTION_TITLES)
    all_matches = list(re.finditer(all_pattern, text, re.IGNORECASE))

    target_pattern = '|'.join(re.escape(title) for title in TARGET_TITLES.get(mode, []))
    target_matches = list(re.finditer(target_pattern, text, re.IGNORECASE))

    if target_matches:
        print("Выбран обычный парсинг")
        for match in target_matches:
            start = match.start()
            title = match.group().strip()
            is_last_section = title.lower() == ALL_SECTION_TITLES[-1].lower()
            if is_last_section:
                sections[title] = text[start:].strip()
                print(f"Раздел '{title}' (последний) извлечен")
                continue
            next_section = next((m for m in all_matches if m.start() > start), None)
            if next_section:
                end = next_section.start()
                sections[title] = text[start:end].strip()
                print(f"Раздел '{title}' извлечен (конец: {next_section.group().strip()})")
            else:
                print(f"Раздел '{title}' найден, но конец не определен. Переход к нейросети.")
                neural_text = await extract_requirements(text, mode)
                if neural_text:
                    sections[title] = neural_text

    else:
        print("Выбран нейросетевой парсинг")
        if mode == "structural":
            structural_text = await extract_requirements(text, mode)
            if structural_text:
                sections["Конструктивные решения"] = structural_text
        elif mode == "arch":
            arch_text = await extract_requirements(text, mode)
            if arch_text:
                sections["Архитектурно-строительные решения"] = arch_text
        elif mode == "engineer":
            engineer_text = await extract_requirements(text, mode)
            if engineer_text:
                sections["Инженерные системы"] = engineer_text
    return sections


async def count_tokens(text: str) -> int:
    """Асинхронная версия подсчета токенов"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_count_tokens, text)

def _sync_count_tokens(text: str) -> int:
    """Синхронная реализация подсчета токенов"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


async def extract_text_from_pdf(pdf_path: str) -> str:
    """Асинхронная обёртка для синхронного извлечения текста из PDF"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_extract_text_from_pdf, pdf_path)


def _sync_extract_text_from_pdf(pdf_path: str) -> str:
    """Синхронная реализация извлечения текста из PDF"""
    reader = PdfReader(pdf_path)
    text = []
    for page_num, page in enumerate(reader.pages, start=1):
        content = page.extract_text() or ""
        text.append(f"\n=== НАЧАЛО СТРАНИЦЫ {page_num} ===\n{content}\n=== КОНЕЦ СТРАНИЦЫ {page_num} ===")
    return "\n".join(text)


async def split_text(text: str, max_tokens: int = 28000, overlap: int = 1000) -> list[str]:
    """Асинхронная версия разделения текста"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_split_text, text, max_tokens, overlap)

def _sync_split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Разделяет текст на чанки с перекрытием, сохраняя информацию о страницах."""
    tokenizer = tiktoken.get_encoding("cl100k_base")

    pages = []
    current_page = None
    for line in text.split('\n'):
        if line.startswith('=== НАЧАЛО СТРАНИЦЫ') or line.startswith('=== СТРАНИЦА'):
            parts = line.split()
            if current_page is not None:
                pages.append(current_page)
            try:
                page_num = int(parts[3] if parts[1] == 'НАЧАЛО' else parts[2])
                current_page = {'num': page_num, 'text': ''}
            except (IndexError, ValueError):
                continue

        elif current_page is not None and not line.startswith('=== КОНЕЦ СТРАНИЦЫ'):
            current_page['text'] += line + '\n'
    if (current_page is not None):
        pages.append(current_page)

    chunks = []
    current_chunk = ""
    current_chunk_pages = set()
    current_tokens = 0

    for page in pages:
        page_text = f"\n\n=== СТРАНИЦА {page['num']} ===\n{page['text']}"
        page_tokens = len(tokenizer.encode(page_text))

        if current_tokens + page_tokens > max_tokens:
            if current_chunk:
                chunk_header = f"Страницы: {', '.join(map(str, sorted(current_chunk_pages)))}\n"
                chunks.append(chunk_header + current_chunk)

            current_chunk = ""
            current_chunk_pages = set()
            current_tokens = 0

            if overlap > 0 and chunks:
                prev_text = chunks[-1]
                overlap_text = tokenizer.decode(tokenizer.encode(prev_text)[-overlap:])
                current_chunk = overlap_text + "\n"
                current_tokens = overlap

        current_chunk += page_text
        current_chunk_pages.add(page['num'])
        current_tokens += page_tokens

    if current_chunk:
        chunk_header = f"Страницы: {', '.join(map(str, sorted(current_chunk_pages)))}\n"
        chunks.append(chunk_header + current_chunk)

    return [chunk for chunk in chunks if chunk.strip()]


async def create_vector_db(text_chunks):
    """Асинхронная обёртка для создания векторной БД"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_create_vector_db, text_chunks)

def _sync_create_vector_db(text_chunks):
    """Создаёт векторную базу данных для семантического поиска."""
    if not text_chunks:
        raise ValueError("Не получено данных для обработки")

    valid_chunks = [chunk for chunk in text_chunks if chunk.strip()]

    if not valid_chunks:
        raise ValueError("Все чанки пусты. Невозможно создать векторную базу")

    try:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        return FAISS.from_texts(valid_chunks, embeddings)
    except Exception as e:
        print(f"Ошибка создания векторной базы: {e}")
        raise


async def find_similar_sections(query: str, vector_db, k: int = 3):
    """Асинхронная версия поиска похожих разделов"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_find_similar_sections, query, vector_db, k)


def _sync_find_similar_sections(query: str, vector_db, k: int):
    """Находит похожие разделы в векторной базе данных."""
    return vector_db.similarity_search(query, k=k)


async def compare_documents(technical_spec, result_doc, mode: str = "arch"):
    """Сравнивает разделы ТЗ и результата с учетом типа анализа."""
    mode_prompts = {
        "arch": {
            "system": "Анализ архитектурных решений. Особое внимание удели материалам стен и полов, толщине стен.",
            "user": "Особое внимание удели всем материалам стен и полов, а также толщине стен"
        },
        "structural": {
            "system": "Анализ конструктивных решений. Проверь соответствие материалов, армирования, несущих конструкций.",
            "user": "Особое внимание удели соответствию несущих конструкций, армирования, материалов по прочностным характеристикам"
        },
        "engineer": {
            "system": "Анализ инженерных систем. Проверь соответствие оборудования, производителей, технических характеристик.",
            "user": "Особое внимание удели соответствию указанного оборудования (отопление, вентиляция), производителей, технических параметров"
        }
    }

    prompt = mode_prompts.get(mode, mode_prompts["arch"])

    messages = [
        {"role": "system",
         "content": f"Ты эксперт по анализу документов. {prompt['system']} При анализе обязательно указывай номера страниц, на которых найдены несоответствия."},
        {"role": "user", "content": f"Вот часть технического задания:\n{technical_spec}\n\n"
                                    f"Вот соответствующий результат работы:\n{result_doc}\n\n"
                                    "Найди несоответствия и укажи, что выполнено правильно, а что — нет. "
                                    "Обязательно указывай номера страниц, на которых найдены проблемы (И ИЗ ТЗ И ИЗ РЕЗУЛЬТАТА РАБОТЫ)"
                                    f"{prompt['user']}"}
    ]

    retries = 3
    while retries > 0:
        current_key = await key_manager.get_key()
        if not current_key:
            await asyncio.sleep(1)
            print("Все ключи исчерпаны")
            return ""
        try:
            print("Отправлен запрос")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"},
                    json={"model": "deepseek/deepseek-chat:free", "messages": messages},
                    timeout=aiohttp.ClientTimeout(total=150)
                ) as response:
                    await key_manager.update_usage(current_key, response)
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        await asyncio.sleep(1)
                        print(f"Ключ {current_key[-5:]}... исчерпан")
                        await key_manager.handle_error(current_key)
                        retries -= 1
                        continue
                    else:
                        await asyncio.sleep(1)
                        print(f"Ошибка {response.status} с ключом {current_key[-5:]}...")
                        return ""
        except Exception as e:
            await asyncio.sleep(1)
            print(f"Ошибка с ключом {current_key[-5:]}...: {str(e)}")
            await key_manager.handle_error(current_key)
            retries -= 1
    return ""


async def process_chunk(result_chunk, technical_db, mode: str = "arch"):
    """Обрабатывает чанк результата и сравнивает с похожими разделами ТЗ."""
    result_pages = set()
    for line in result_chunk.split('\n'):
        if line.startswith('=== СТРАНИЦА'):
            result_pages.add(int(line.split()[2]))
        elif line.startswith('=== НАЧАЛО СТРАНИЦЫ'):
            result_pages.add(int(line.split()[2]))

    similar_sections = await find_similar_sections(result_chunk, technical_db)
    tasks = [compare_documents(section, result_chunk, mode) for section in similar_sections]
    results = await asyncio.gather(*tasks)

    if result_pages:
        pages_sorted = sorted(result_pages)
        if len(pages_sorted) > 1:
            page_info = f"\n(Анализируемые страницы результата: {pages_sorted[0]}-{pages_sorted[-1]})"
        else:
            page_info = f"\n(Анализируемая страница результата: {pages_sorted[0]})"

        results = [r + page_info for r in results if not r.strip().endswith(')')]

    return results


async def split_text_summarized(text: str, max_tokens: int = 13000) -> list[str]:
    """Асинхронная версия разделения текста"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_split_text_summarized, text, max_tokens)

def _sync_split_text_summarized(text: str, max_tokens: int) -> list[str]:
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
                "Не нужно писать рекомендации, сфокусируйся на недочётах"
            ).format(report_part=report_part)
        }
    ]
    retries = 3
    while retries > 0:
        current_key = await key_manager.get_key()
        if not current_key:
            await asyncio.sleep(1)
            print("Все ключи исчерпаны")
            return ""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"},
                    json={"model": "deepseek/deepseek-chat:free", "messages": messages},
                    timeout=aiohttp.ClientTimeout(total=150)
                ) as response:
                    await key_manager.update_usage(current_key, response)

                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        await asyncio.sleep(1)
                        print(f"Ключ {current_key[-5:]}... исчерпан")
                        await key_manager.handle_error(current_key)
                        retries -= 1
                        continue
                    else:
                        await asyncio.sleep(1)
                        print(f"Ошибка {response.status} с ключом {current_key[-5:]}...")
                        return ""
        except Exception as e:
            await asyncio.sleep(1)
            print(f"Ошибка с ключом {current_key[-5:]}...: {str(e)}")
            await key_manager.handle_error(current_key)
            retries -= 1
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
                "Не нужно писать рекомендации, сфокусируйся на недочётах"
            ).format(combined_report=combined_report)
        }
    ]

    retries = 3
    while retries > 0:
        current_key = await key_manager.get_key()
        if not current_key:
            print("Все ключи исчерпаны")
            return ""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"},
                    json={"model": "deepseek/deepseek-chat:free", "messages": messages},
                    timeout=aiohttp.ClientTimeout(total=150)
                ) as response:
                    await key_manager.update_usage(current_key, response)

                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        await asyncio.sleep(1)
                        print(f"Ключ {current_key[-5:]}... исчерпан")
                        await key_manager.handle_error(current_key)
                        retries -= 1
                        continue
                    else:
                        await asyncio.sleep(1)
                        print(f"Ошибка {response.status} с ключом {current_key[-5:]}...")
                        return ""
        except Exception as e:
            await asyncio.sleep(1)
            print(f"Ошибка с ключом {current_key[-5:]}...: {str(e)}")
            await key_manager.handle_error(current_key)
            retries -= 1
    return ""

async def async_write_file(filename: str, content: str):
    async with aiofiles.open(filename, "w", encoding="utf-8") as f:
        await f.write(content)


async def analyze_documents(category: str, tz_path: str, result_path: str, user_id: int) -> str:
    print('Загрузка текстов')
    technical_text, result_text = await asyncio.gather(
        extract_text_from_pdf(tz_path),
        extract_text_from_pdf(result_path)
    )

    mode_map = {
        'arch_category': ('arch', 'Архитектурно-строительные решения'),
        'constr_category': ('structural', 'Конструктивные решения'),
        'eng_category': ('engineer', 'Инженерные системы'),
        "water_supply_category": ("water_supply", "Система водоснабжения"),
        "water_drain_category": ("water_drain", "Система водоотведения"),
        "heat_network_category": ("heat_network", "Тепловые сети и ИТП"),
        "hvac_category": ("hvac", "Отопление, вентиляция и кондиционирование")
    }
    mode, section_name = mode_map.get(category, ('arch', 'Архитектурно-строительные решения'))

    print('Получение требований из ТЗ')
    ts_sections = await extract_sections_from_ts(technical_text, mode=mode)

    technical_section_text = ts_sections.get(section_name)
    print("Первые 100 символов раздела:\n", technical_section_text[:100])
    print("Последние 100 символов раздела:\n", technical_section_text[-100:])

    technical_tokens, result_tokens = await asyncio.gather(
        count_tokens(technical_section_text),
        count_tokens(result_text)
    )

    print(f"Токенов в техническом задании: {technical_tokens}")
    print(f"Токенов в результате: {result_tokens}")

    print('Разбиение текстов на части')
    technical_chunks, result_chunks = await asyncio.gather(
        split_text(technical_section_text),
        split_text(result_text)
    )
    print(f"Количество чанков технического задания: {len(technical_chunks)}")
    print('Создание векторных хранилищ')
    technical_db = await create_vector_db(technical_chunks)

    print('Сбор всех результатов сравнения')
    all_results = []

    semaphore = asyncio.Semaphore(3)

    async def process_chunk_with_semaphore(chunk):
        async with semaphore:
            return await process_chunk(chunk, technical_db, mode)

    tasks = [process_chunk_with_semaphore(chunk) for chunk in result_chunks]
    for task in tqdm_asyncio.as_completed(tasks, desc="Обработка документов"):
        results = await task
        all_results.extend(results)
        await asyncio.sleep(0.1)

    unstructured_report = "\n\n".join(all_results)
    report_parts = await split_text_summarized(unstructured_report)
    print(f"Отчёт разделён на {len(report_parts)} частей.")
    if len(report_parts) > 1:
        structured_parts = await asyncio.gather(
            *(structure_report_part(part) for part in report_parts))
        final_report = await final_structure_report(structured_parts)
    else:
        final_report = await final_structure_report(report_parts)
    report_filename = f"temp/report_{user_id}_{int(datetime.now().timestamp())}.txt"
    await async_write_file(report_filename, final_report)
    return report_filename


async def generate_answer(file_path: str, question: str) -> str:
    try:

        raw_text = await extract_text_from_pdf(file_path)
        chunks = await split_text(raw_text, max_tokens=8000, overlap=500)

        vector_db = await create_vector_db(chunks)

        relevant_docs = await find_similar_sections(question, vector_db, k=3)

        tokenizer = tiktoken.get_encoding("cl100k_base")
        context = []
        total_tokens = len(tokenizer.encode(question)) + 500

        for doc in relevant_docs:
            doc_content = doc.page_content
            doc_tokens = len(tokenizer.encode(doc_content))

            if total_tokens + doc_tokens > 26000:
                break

            context.append(doc_content)
            total_tokens += doc_tokens

        messages = [
            {
                "role": "system",
                "content": "Ты эксперт по анализу документов. Ответь на вопрос используя ТОЛЬКО предоставленные фрагменты документа. "
                           "Если ответа нет в документах, скажи об этом. Указывай номера страниц в формате [Страница X]."
            },
            {
                "role": "user",
                "content": f"Контекст:\n{'\n\n'.join(context)}\n\nВопрос: {question}. ОБЯЗАТЕЛЬНО УКАЖИ СТРАНИЦУ НА КОТОРОЙ НАШЁЛ ОТВЕТ."
            }
        ]
        retries = 3
        while retries > 0:
            current_key = await key_manager.get_key()
            if not current_key:
                print("Все ключи исчерпаны")
                return "Произошла ошибка: все ключи API исчерпаны"

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {current_key}"},
                        json={
                            "model": "deepseek/deepseek-chat:free",
                            "messages": messages,
                            "temperature": 0.3,
                            "max_tokens": 2000
                        }
                    ) as response:
                        await key_manager.update_usage(current_key, response)

                        if response.status == 200:
                            result = await response.json()
                            return result['choices'][0]['message']['content']
                        elif response.status == 429:
                            await asyncio.sleep(1)
                            print(f"Ключ {current_key[-5:]}... исчерпан")
                            await key_manager.handle_error(current_key)
                            retries -= 1
                            continue
                        else:
                            await asyncio.sleep(1)
                            error_text = await response.text()
                            print(f"API Error {response.status}: {error_text[:200]}")
                            return "Произошла ошибка при обработке запроса"
            except Exception as e:
                await asyncio.sleep(1)
                print(f"Ошибка с ключом {current_key[-5:]}...: {str(e)}")
                await key_manager.handle_error(current_key)
                retries -= 1
        return "Произошла внутренная ошибка"
    except Exception as e:
        await asyncio.sleep(1)
        print(f"Unexpected error: {str(e)}")
        return "Произошла внутренняя ошибка"
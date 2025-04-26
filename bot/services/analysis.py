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
from services.key_manager import key_manager
from config import OPENAI_API_KEY
from services.utils import generate_short_uuid
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def extract_requirements(text: str, mode: str) -> str:
    """Извлекает инженерные требования по частям"""
    chunks = await split_text(text, max_tokens=60000, overlap=2000)
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
                try:
                    completion = await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        timeout=150
                    )

                    return completion.choices[0].message.content.strip()

                except Exception as e:
                    await asyncio.sleep(1)
                    print(f"Ошибка при запросе к OpenAI: {str(e)}")
                    retries -= 1
                    if retries == 0:
                        return None

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
        "Раздел «Системы электроснабжения»"
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

            page_start = text.rfind('=== НАЧАЛО СТРАНИЦЫ', 0, start)
            if page_start != -1:
                # Находим номер этой страницы
                page_num_start = page_start + len('=== НАЧАЛО СТРАНИЦЫ ')
                page_num_end = text.find(' ===', page_num_start)
                if page_num_end != -1:
                    page_num = text[page_num_start:page_num_end].strip()
                    # Добавляем маркер начала страницы в начало раздела
                    section_start = f"=== НАЧАЛО СТРАНИЦЫ {page_num} ===\n"
                else:
                    section_start = ""
            else:
                section_start = ""

            if is_last_section:
                sections[title] = text[start:].strip()
                print(f"Раздел '{title}' (последний) извлечен")
                continue
            next_section = next((m for m in all_matches if m.start() > start), None)
            if next_section:
                end = next_section.start()
                sections[title] = section_start + text[start:end].strip()
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
        elif mode == "water_supply":
            water_supply_text = await extract_requirements(text, mode)
            if water_supply_text:
                sections["Водоснабжение и канализация"] = water_supply_text
        elif mode == "water_drain":
            water_drain_text = await extract_requirements(text, mode)
            if water_drain_text:
                sections["Водоснабжение и канализация"] = water_drain_text
        elif mode == "heat_network":
            heat_network_text = await extract_requirements(text, mode)
            if heat_network_text:
                sections["Отопление, вентиляция и кондиционирование"] = heat_network_text
        elif mode == "hvac":
            hvac_text = await extract_requirements(text, mode)
            if hvac_text:
                sections["Отопление, вентиляция и кондиционирование"] = hvac_text
        else:
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
        if page_num <= len(reader.pages):
            text.append(f"\n=== НАЧАЛО СТРАНИЦЫ {page_num} ===\n{content}\n=== КОНЕЦ СТРАНИЦЫ {page_num} ===")
    return "\n".join(text)


async def split_text(text: str, max_tokens: int = 80000, overlap: int = 5000) -> list[str]:
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
        elif line.startswith('=== КОНЕЦ СТРАНИЦЫ'):
            if current_page is not None:
                current_page['text'] += line + '\n'
                pages.append(current_page)
                current_page = None

        elif current_page is not None:
            current_page['text'] += line + '\n'
    if current_page is not None:
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
        return FAISS.from_texts(
            sorted(valid_chunks, key=lambda x: hash(x)),
            embeddings)
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
            "system": "Анализ архитектурных решений."
                      "Если производитель или материал совпадает - несоответствия нет. "
                      "Если производитель или материал, но в ТЗ есть пометка о том, что аналог из результата можно использовать - несоответствия нет. "
                      "Если производитель или материал и аналог не допускается - укажи как несоответствие.",
            "user": "Проверь соответствие архитектурных требований и реализации. Особое внимание удели всем материалам стен и полов, толщине стен и т. д."
        },
        "structural": {
            "system": "Анализ конструктивных решений."
                      "Если производитель или материал совпадает - несоответствия нет. "
                      "Если производитель или материал отличается, но в ТЗ есть пометка о том, что аналог из результата можно использовать - несоответствия нет. "
                      "Если производитель или материал отличается и аналог не допускается - укажи как несоответствие.",
            "user": "Проверь соответствие конструктивных требований и реализации. Особое внимание удели соответствию несущих конструкций, армирования, материалов по прочностным характеристикам и т. д."
        },
        "engineer": {
            "system": "Анализ инженерных систем."
                      "Если производитель или материал совпадает - несоответствия нет. "
                    "Если производитель или материал отличается, но в ТЗ есть пометка о том, что аналог из результата можно использовать - несоответствия нет. "
                    "Если производитель или материал отличается и аналог не допускается - укажи как несоответствие.",
            "user": "Проверь соответствие инженерных требований и реализации. Особое внимание удели соответствию указанного оборудования (отопление, вентиляция), производителей, технических параметров и т. д."
        }
    }

    prompt = mode_prompts.get(mode, mode_prompts["engineer"])
    total_tokens = await count_tokens(str(technical_spec)) + await count_tokens(str(result_doc))
    messages = [
        {"role": "system",
         "content": f"Ты эксперт по анализу документов. {prompt['system']} При анализе обязательно указывай номера страниц, на которых найдены несоответствия."},
        {"role": "user", "content": "Найди несоответствия и укажи, что выполнено правильно, а что — нет. "
                                    "Обязательно указывай номера страниц, на которых найдены проблемы (И ИЗ ТЗ И ИЗ РЕЗУЛЬТАТА РАБОТЫ), ТОЛЬКО ОБЯЗАТЕЛЬНО УБЕДИСЬ ЧТО НЕ ПЕРЕПУТАЕШЬ ИХ, ЧТОБЫ НЕ БЫЛО ТАКОГО, ЧТО ТЫ УКАЗАЛ СТРАНИЦУ ТЗ = 61 ХОТЯ ИХ В ТЗ МЕНЬШЕ"
                                    "При указывании страниц ориентируйся на конструкции '=== НАЧАЛО СТРАНИЦЫ {page_num} === И === КОНЕЦ СТРАНИЦЫ {page_num} ==='"
                                    "ЕСЛИ ЕСТЬ НЕСООТВЕТСТВИЯ ОБЯЗАТЕЛЬНО ГОВОРИ КАК ТРЕБУЕТ ТЗ, И КАК СДЕЛАНО В РЕЗУЛЬТАТЕ, ЧТОБЫ БЫЛО ПОНЯТНО ЧТО ИМЕННО НЕ ТАК"
                                    f"{prompt['user']}"
                                    f"Вот часть технического задания:\n{technical_spec}\n\n"
                                    f"Вот соответствующий результат работы:\n{result_doc}\n\n"
                                    }
    ]
    print(total_tokens)
    #print(messages)
    retries = 3
    while retries > 0:
        try:
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                timeout=150
            )

            return completion.choices[0].message.content

        except Exception as e:
            await asyncio.sleep(1)
            print(f"Ошибка при запросе к OpenAI: {str(e)}")
            retries -= 1
            if retries == 0:
                return ""
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
                f"Вот часть отчёта по анализу документов:\n{report_part}\n\n"
                "Структурируй его в следующем строгом формате:\n"
                "### Раздел 1: Ключевые несоответствия\n"
                "#### Подраздел: [Название конструкции/системы]\n"
                "- Пункт 1: [Описание несоответствия] (ТЗ стр.X / РР стр.Y)\n"
                "- Пункт 2: [Описание несоответствия] (ТЗ стр.X / РР стр.Y)\n\n"
                "### Раздел 2: Технические ошибки\n"
                "- Ошибка 1: [Описание] (ТЗ стр.X / РР стр.Y)\n"
                "- Ошибка 2: [Описание] (ТЗ стр.X / РР стр.Y)\n\n"
                "### Раздел 3: Нарушения нормативных требований\n"
                "- Нарушение 1: [Описание] (Норматив: (номер норматива, если знаешь), ТЗ стр.X / РР стр.Y)\n\n"
                "ТРЕБОВАНИЯ К ФОРМАТУ:\n"
                "1. Каждый пункт должен начинаться с дефиса\n"
                "2. Обязательно указывать страницы из ТЗ и Результата работ\n"
                "3. Не использовать Markdown кроме ### для разделов и #### для подразделов\n"
                "НЕ УДАЛЯЙ и не изменяй формулировки из исходного отчёта.\n"
                "КАЖДОЕ НЕСООТВЕТСТВИЕ ДОЛЖНО ССЫЛАТЬСЯ НА СТРАНИЦУ И ИЗ ТЗ И ИЗ РЕЗУЛЬТА РАБОТЫ\n"

                "Не добавляй новую информацию."
                "Не нужно писать рекомендации, сфокусируйся на недочётах"
            )
        }
    ]
    retries = 3
    while retries > 0:
        try:
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                timeout=150
            )

            return completion.choices[0].message.content

        except Exception as e:
            await asyncio.sleep(1)
            print(f"Ошибка при запросе к OpenAI: {str(e)}")
            retries -= 1
            if retries == 0:
                return ""
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
                f"Вот объединённый отчёт:\n{combined_report}\n\n"
                "Структурируй его в следующем строгом формате:\n"
                "### Раздел 1: Ключевые несоответствия\n"
                "#### Подраздел: [Название конструкции/системы]\n"
                "- Пункт 1: [Описание несоответствия] (ТЗ стр.X / РР стр.Y)\n"
                "- Пункт 2: [Описание несоответствия] (ТЗ стр.X / РР стр.Y)\n\n"
                "### Раздел 2: Технические ошибки\n"
                "- Ошибка 1: [Описание] (ТЗ стр.X / РР стр.Y)\n"
                "- Ошибка 2: [Описание] (ТЗ стр.X / РР стр.Y)\n\n"
                "### Раздел 3: Нарушения нормативных требований\n"
                "- Нарушение 1: [Описание] (Норматив: (номер норматива, если знаешь), ТЗ стр.X / РР стр.Y)\n\n"
                "ТРЕБОВАНИЯ К ФОРМАТУ:\n"
                "1. Каждый пункт должен начинаться с дефиса\n"
                "2. Обязательно указывать страницы из ТЗ и Результата работ\n"
                "3. Не использовать Markdown кроме ### для разделов и #### для подразделов\n"
                "НЕ УДАЛЯЙ и не изменяй формулировки из исходного отчёта.\n"
                "КАЖДОЕ НЕСООТВЕТСТВИЕ ДОЛЖНО ССЫЛАТЬСЯ НА СТРАНИЦУ И ИЗ ТЗ И ИЗ РЕЗУЛЬТА РАБОТЫ\n"
                "Не нужно писать рекомендации, сфокусируйся на недочётах"
            )
        }
    ]

    retries = 3
    while retries > 0:
        try:
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                timeout=150
            )

            return completion.choices[0].message.content

        except Exception as e:
            await asyncio.sleep(1)
            print(f"Ошибка при запросе к OpenAI: {str(e)}")
            retries -= 1
            if retries == 0:
                return ""
    return ""

async def async_write_file(filename: str, content: str):
    async with aiofiles.open(filename, "w", encoding="utf-8") as f:
        await f.write(content)


async def analyze_documents(category: str, tz_path: str, result_path: str) -> str:
    print('Загрузка текстов')
    technical_text, result_text = await asyncio.gather(
        extract_text_from_pdf(tz_path),
        extract_text_from_pdf(result_path)
    )

    mode_map = {
        'arch_category': ('arch', 'Архитектурно-строительные решения'),
        'constr_category': ('structural', 'Конструктивные решения'),
        'eng_category': ('engineer', 'Инженерные системы'),
        "water_supply_category": ("water_supply", "Водоснабжение и канализация"),
        "water_drain_category": ("water_drain", "Водоснабжение и канализация"),
        "heat_network_category": ("heat_network", "Отопление, вентиляция и кондиционирование"),
        "hvac_category": ("hvac", "Отопление, вентиляция и кондиционирование")
    }

    report_name_map = {
        'arch_category': 'Архитектурно-строительные решения',
        'constr_category': 'Конструктивные решения',
        'eng_category': 'Инженерные системы (общий)',
        "water_supply_category": "Система водоснабжения",
        "water_drain_category": "Система водоотведения",
        "heat_network_category": "Тепловые сети и ИТП",
        "hvac_category": "Отопление вентиляция и кондиционирование"
    }

    mode, section_name = mode_map.get(category, ('arch', 'Архитектурно-строительные решения'))

    report_name = report_name_map.get(category, 'Архитектурно-строительные решения')

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

    short_uuid = await generate_short_uuid()

    report_filename = f"temp/{report_name}_{datetime.now().strftime("%d_%m_%Y")}_{short_uuid}.txt"
    await async_write_file(report_filename, final_report)
    return report_filename


async def generate_answer(file_path: str, question: str) -> str:
    try:
        raw_text = await extract_text_from_pdf(file_path)
        print("Извлечённый текст")
        print(len(raw_text))
        chunks = await split_text(raw_text, max_tokens=34000, overlap=1000)
        print(len(chunks))
        vector_db = await create_vector_db(chunks)

        relevant_docs = await find_similar_sections(question, vector_db, k=3)

        tokenizer = tiktoken.get_encoding("cl100k_base")
        context = []
        total_tokens = len(tokenizer.encode(question)) + 500

        for doc in relevant_docs:
            doc_content = doc.page_content
            doc_tokens = len(tokenizer.encode(doc_content))

            if total_tokens + doc_tokens > 105000:
                break

            context.append(doc_content)
            total_tokens += doc_tokens

        messages = [
            {
                "role": "system",
                "content": """Ты эксперт по анализу строительной документации. Анализируй вопросы строго по правилам:
        
                1. ЕСЛИ ВОПРОС:
                - Слишком общий ("Какие материалы?")
                - Не содержит конкретики ("Что тут написано?")
                - Не относится к документу ("Как погода?")
                - Непонятен или бессмыслен ("Абракадабра")
                ТО Ответ: "Вопрос слишком общий или не относится к документации."
                
                2. ЕСЛИ ВОПРОС В ЦЕЛОМ КОРРЕКТЕН:
                - Используй ТОЛЬКО предоставленные фрагменты
                - Указывай страницы в формате [Страница X]
                - Если ответа нет, то "Информация не найдена в документах"
                
                3. ОСОБЫЕ СЛУЧАИ:
                - При запросе сравнений → укажи оба источника
                - При технических терминах → давай точные формулировки из текста"""
            },
            {
                "role": "user",
                "content": f"Контекст документа:\n{'\n\n'.join(context)}\n\nВопрос: {question}. Действуй строго по инструкции выше. Не придумывай ответы!"
            }
        ]
        retries = 3
        while retries > 0:
            try:
                client = AsyncOpenAI(api_key=OPENAI_API_KEY)

                completion = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                    timeout=150
                )

                return completion.choices[0].message.content
            except Exception as e:
                await asyncio.sleep(1)
                print(f"Ошибка при запросе: {str(e)}")
                retries -= 1
                if retries == 0:
                    return "Произошла ошибка при обработке запроса"
        return "Произошла внутренняя ошибка: превышено количество попыток"
    except Exception as e:
        await asyncio.sleep(1)
        print(f"Unexpected error: {str(e)}")
        return "Произошла внутренняя ошибка"
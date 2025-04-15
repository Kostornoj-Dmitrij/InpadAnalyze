import os
import re

import pandas as pd
from aiogram import Bot
from aiogram.types import Message
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
font_path = os.path.join(project_dir, 'fonts/DejaVuSans.ttf')
pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))

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


async def convert_to_pdf(txt_path: str) -> str:
    pdf_path = txt_path.replace('.txt', '.pdf')

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Normal_RU',
        fontName='DejaVuSans',
        fontSize=12,
        leading=14,
        parent=styles['Normal']
    ))
    styles.add(ParagraphStyle(
        name='Heading2_RU',
        fontName='DejaVuSans',
        fontSize=14,
        leading=16,
        parent=styles['Heading2']
    ))

    story = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('##'):
                p = Paragraph(line.replace('#', ''), styles['Heading2_RU'])
            else:
                p = Paragraph(line, styles['Normal_RU'])
            story.append(p)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    doc.build(story)

    return pdf_path


async def convert_to_xls(txt_path: str) -> str:
    xls_path = txt_path.replace('.txt', '.xlsx')

    data = {
        'Раздел': [],
        'Подраздел': [],
        'Тип проблемы': [],
        'Описание': [],
        'Страница ТЗ': [],
        'Страница РР': []
    }

    current_section = ""
    current_subsection = ""

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('### Раздел'):
                current_section = line.replace('###', '').strip()
                current_subsection = ""
            elif line.startswith('#### '):
                current_subsection = line.replace('####', '').strip()

            elif line.startswith('-'):
                item = line[1:].strip()
                description = re.sub(r'^(Ошибка \d+|Нарушение \d+):\s*', '', item)
                tz_page, rr_page = "", ""

                bracket_pairs = []
                stack = []
                for i, char in enumerate(item):
                    if char == '(':
                        stack.append(i)
                    elif char == ')' and stack:
                        start = stack.pop()
                        bracket_pairs.append((start, i))
                if bracket_pairs:
                    last_bracket_start, last_bracket_end = bracket_pairs[-1]
                    pages_text = item[last_bracket_start + 1:last_bracket_end]

                    tz_match = re.search(r'ТЗ\s*стр\.\s*(\d+(?:-\d+)?)', pages_text)
                    rr_match = re.search(r'РР\s*стр\.\s*(\d+(?:-\d+)?)', pages_text)

                    if tz_match:
                        tz_page = tz_match.group(1)
                    if rr_match:
                        rr_page = rr_match.group(1)

                if current_section.startswith("Раздел 1"):
                    problem_type = "Несоответствие"
                elif current_section.startswith("Раздел 2"):
                    problem_type = "Ошибка"
                elif current_section.startswith("Раздел 3"):
                    problem_type = "Нарушение"
                else:
                    problem_type = ""

                data['Раздел'].append(current_section)
                data['Подраздел'].append(current_subsection if current_subsection else "Общее")
                data['Тип проблемы'].append(problem_type)
                data['Описание'].append(description)
                data['Страница ТЗ'].append(tz_page)
                data['Страница РР'].append(rr_page)

    df = pd.DataFrame(data)

    df.fillna("", inplace=True)

    df['Номер раздела'] = df['Раздел'].str.extract(r'Раздел (\d+)').astype(int)
    df.sort_values(['Номер раздела', 'Подраздел'], inplace=True)
    df.drop('Номер раздела', axis=1, inplace=True)

    with pd.ExcelWriter(xls_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Отчет')

        workbook = writer.book
        worksheet = writer.sheets['Отчет']

        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })

        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        for i, col in enumerate(df.columns):
            max_len = max((
                df[col].astype(str).str.len().max(),
                len(col)
            )) + 2
            worksheet.set_column(i, i, min(max_len, 50))

        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

        format_discrepancy = workbook.add_format({'bg_color': '#FFF2CC'})
        format_mistake = workbook.add_format({'bg_color': '#FCE4D6'})
        format_violation = workbook.add_format({'bg_color': '#F2F2F2'})

        for row in range(1, len(df) + 1):
            problem_type = df.at[row - 1, 'Тип проблемы']
            if problem_type == "Несоответствие":
                worksheet.set_row(row, None, format_discrepancy)
            elif problem_type == "Ошибка":
                worksheet.set_row(row, None, format_mistake)
            elif problem_type == "Нарушение":
                worksheet.set_row(row, None, format_violation)

    return xls_path
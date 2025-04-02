from aiosqlite import connect, Row
import os
from contextlib import asynccontextmanager

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @asynccontextmanager
    async def get_connection(self):
        """Контекстный менеджер для получения соединения с базой данных."""
        db = await connect(self.db_path)
        db.row_factory = Row
        try:
            yield db
        finally:
            await db.close()

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_file = os.path.join(project_dir, 'data', 'database.db')

db = Database(db_file)
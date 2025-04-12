from bot.services.key_manager import key_manager
from bot.config import OpenRouter_API_KEYS
from datetime import datetime, timedelta
import asyncio

async def exhaust_keys():
    async with key_manager.lock:
        for key in OpenRouter_API_KEYS:
            key_manager.usage[key] = 50
            key_manager.failed_keys.add(key)

    print("После исчерпания:")
    for status in await key_manager.get_key_statuses():
        print(f"{status['key']}: used {status['used']}, status: {status['status']}")

    key = await key_manager.get_key()
    print(f"Попытка получить ключ когда все исчерпаны: {'успех' if key else 'провал'}")


async def test_reset():
    print("\n=== Перед сбросом ===")
    print(f"Текущий last_reset: {key_manager.last_reset}")
    for status in await key_manager.get_key_statuses():
        print(f"{status['key']}: used {status['used']}, status: {status['status']}")

    # Устанавливаем "вчерашнюю" дату
    key_manager.last_reset = datetime.now() - timedelta(days=1)
    print(f"\nУстановлен last_reset: {key_manager.last_reset}")

    # Принудительно вызываем сброс
    async with key_manager.lock:
        key_manager.usage.clear()
        key_manager.failed_keys.clear()

    print("\nПосле сброса счетчиков:")
    print(f"Текущий last_reset: {key_manager.last_reset}")
    for status in await key_manager.get_key_statuses():
        print(f"{status['key']}: used {status['used']}, status: {status['status']}")
async def test_key_manager():
    print("Текущие статусы ключей:")
    for status in await key_manager.get_key_statuses():
        print(f"{status['key']}: used {status['used']}, status: {status['status']}")

    print("\nПолучаем ключ...")
    key = await key_manager.get_key()
    print(f"Получен ключ: {key[:5]}...{key[-5:] if key else None}")


async def main():
    await test_key_manager()
    await exhaust_keys()
    await test_reset()

if __name__ == "__main__":
    print("=== Запуск тестов менеджера ключей ===")
    asyncio.run(main())
    print("\n=== Все тесты завершены ===")
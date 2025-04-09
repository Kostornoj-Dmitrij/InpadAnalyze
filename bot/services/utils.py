import os

async def cleanup_temp_files(*paths):
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Error deleting file {path}: {e}")
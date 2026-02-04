import os
import json
from config import Config

class UserManager:
    def __init__(self):
        self.memory_file = Config.MEMORY_FILE
        self._ensure_memory_file()

    def _ensure_memory_file(self):
        if not os.path.exists(Config.MEMORY_DIR):
            os.makedirs(Config.MEMORY_DIR)
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump({"Guest": {"history": []}}, f)

    def get_all_users(self):
        with open(self.memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return list(data.keys())

    def create_user(self, username):
        if not username:
            return False
        with open(self.memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if username not in data:
            data[username] = {"history": []}
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            return True
        return False
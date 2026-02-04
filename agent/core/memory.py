import json
import os
from config import Config

class Memory:
    def __init__(self, user_id):
        self.user_id = user_id
        self.file_path = Config.MEMORY_FILE

    def _load_db(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_db(self, data):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_history(self):
        """Get the user's full modification history for reference by VLM"""
        db = self._load_db()
        return db.get(self.user_id, {}).get("history", [])

    def save_record(self, scene_tags, parameter_state):
        """Save a record of satisfactory modifications"""
        db = self._load_db()
        if self.user_id not in db:
            db[self.user_id] = {"history": []}
        
        record = {
            "scene_tags": scene_tags,
            "parameters": parameter_state
        }
        db[self.user_id]["history"].append(record)
        self._save_db(db)
        return f"Saved preference for scene: {scene_tags}"
# Config.py
import os
import json
from dotenv import load_dotenv

class Config:
    def __init__(self, env_path="3pp_config.env", config_path="config.json"):
        # 加载 LangSmith 的 .env 配置
        load_dotenv(dotenv_path=env_path)

        # 加载模型相关的 config.json
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def get(self, key, default=None):
        """获取配置项，从 config.json 中读取"""
        return self.config.get(key, default)

# 实例化配置对象
config = Config()
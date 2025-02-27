import json
import os

class Config:
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 数据库配置
        db = config["database"]
        self.CONNECTION_STRING = (
            f"postgresql+psycopg2://{db['username']}:{db['password']}@"
            f"{db['host']}:{db['port']}/{db['dbname']}"
        )

        # Ollama 配置
        self.MODEL = config["ollama"]["model"]

# 单例模式实例化
config = Config()

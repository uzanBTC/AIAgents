import unittest
from unsloth import FastLanguageModel
import torch

class MyTestCase(unittest.TestCase):
    def test_something(self):
        model_name = "deepseek-ai/DeepSeek-7B"
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                load_in_4bit=True,
                device_map="auto"
            )
            print("模型加载成功，Unsloth 支持此模型！")
        except Exception as e:
            print(f"加载失败，错误信息：{e}")

    def test_cuda(self):
        print("PyTorch 版本:", torch.__version__)
        print("CUDA 是否可用:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("当前 GPU 设备:", torch.cuda.current_device())
            print("GPU 名称:", torch.cuda.get_device_name(0))
        else:
            print("未检测到 CUDA，可能需要检查驱动或环境配置")

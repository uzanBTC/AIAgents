import os
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

class DeepSeekFinetuner:
    def __init__(self, model_name, dataset_path, output_dir="output_dir", gguf_path="finetuned_deepseek.gguf", modelfile_path="Modelfile.txt"):
        """
        初始化 DeepSeek 微调器

        :param model_name: 预训练模型名称
        :param dataset_path: 微调数据集路径（JSON 格式）
        :param output_dir: 训练输出目录
        :param gguf_path: 导出的 GGUF 模型文件路径
        :param modelfile_path: Modelfile 文件路径
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.gguf_path = gguf_path
        self.modelfile_path = modelfile_path

    def load_model(self):
        """加载并配置模型"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            load_in_4bit=True,  # 启用 4 位量化以节省显存
            device_map="auto"   # 自动分配到 GPU
        )
        # 配置 QLoRA
        model = FastLanguageModel.for_training(
            model,
            lora_config={
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"]
            }
        )
        return model, tokenizer

    def load_dataset(self):
        """加载并处理数据集"""
        dataset = load_dataset("json", data_files=self.dataset_path)
        dataset = dataset.map(lambda x: {"text": f"{x['prompt']} {x['response']}"})
        return dataset

    def train_model(self, model, tokenizer, dataset):
        """配置并启动模型训练"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=10,
            optim="paged_adamw_8bit",
            max_steps=-1
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            dataset_text_field="text",
            max_seq_length=512
        )
        trainer.train()

    def save_gguf(self, model):
        """导出模型为 GGUF 格式"""
        model.save_gguf(self.gguf_path)

    def create_modelfile(self):
        """创建 Modelfile"""
        modelfile_content = f"""FROM {self.gguf_path}
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
"""
        with open(self.modelfile_path, "w") as f:
            f.write(modelfile_content)

    def run(self):
        """执行微调流程"""
        print("加载模型...")
        model, tokenizer = self.load_model()
        print("加载数据集...")
        dataset = self.load_dataset()
        print("开始训练...")
        self.train_model(model, tokenizer, dataset)
        print("训练完成，导出 GGUF 模型...")
        self.save_gguf(model)
        print("创建 Modelfile...")
        self.create_modelfile()
        print(f"微调完成！请使用以下命令部署模型：")
        print(f"ollama create deepseek-ac -f {self.modelfile_path}")
        print(f"ollama run deepseek-ac")

if __name__ == "__main__":
    # 配置参数
    model_name = "deepseek-ai/DeepSeek-7B"
    dataset_path = "dataset.json"
    finetuner = DeepSeekFinetuner(model_name, dataset_path)
    finetuner.run()
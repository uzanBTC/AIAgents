import os
import json
import re

import ollama
from openai import OpenAI  # 导入OpenAI库用于访问GPT模型
from logger import LOG  # 导入日志模块
from config import Config  # 从config模块导入Config类，用于配置管理


class LLM:
    def __init__(self):
        # 创建一个OpenAI客户端实例
        self.client = OpenAI()
        self.config = Config()
        # 从TXT文件加载提示信息
        with open("../prompts/report_prompt.txt", "r", encoding='utf-8') as file:
            self.system_prompt = file.read()

        with open("../prompts/hackernews_system_prompt.txt", "r", encoding='utf-8') as file:
            self.hackernews_system_prompt = file.read()

    def generate_daily_report(self, markdown_content):
        # 使用从TXT文件加载的提示信息
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": markdown_content},
        ]

        if self.config.dry_run:
            # 如果启用了dry_run模式，将不会调用模型，而是将提示信息保存到文件中
            LOG.info("Dry run mode enabled. Saving prompt to file.")
            with open("daily_progress/prompt.txt", "w+") as f:
                # 格式化JSON字符串的保存
                json.dump(messages, f, indent=4, ensure_ascii=False)
            LOG.debug("Prompt已保存到 daily_progress/prompt.txt")

            return "DRY RUN"

        # 日志记录开始生成报告
        LOG.info("使用 GPT 模型开始生成报告。")

        try:
            # 调用OpenAI GPT模型生成报告
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 指定使用的模型版本
                messages=messages
            )
            LOG.debug("GPT response: {}", response)
            # 返回模型生成的内容
            return response.choices[0].message.content
        except Exception as e:
            # 如果在请求过程中出现异常，记录错误并抛出
            LOG.error(f"生成报告时发生错误：{e}")
            raise

    def generate_hackernews_report(self, markdown_content):
        # 使用从TXT文件加载的提示信息
        messages = [
            {"role": "system", "content": self.hackernews_system_prompt},
            {"role": "user", "content": markdown_content},
        ]

        if self.config.dry_run:
            # 如果启用了dry_run模式，将不会调用模型，而是将提示信息保存到文件中
            LOG.info("Dry run mode enabled. Saving prompt to file.")
            os.makedirs("hackernews_reports", exist_ok=True)  # 创建文件夹
            with open("hackernews_reports/prompt.txt", "w+", encoding='utf-8') as f:
                # 格式化JSON字符串的保存
                json.dump(messages, f, indent=4, ensure_ascii=False)
            LOG.debug("Prompt已保存到 hackernews_reports/prompt.txt")

            return "DRY RUN"

        # 日志记录开始生成报告
        LOG.info("使用大模型开始生成 Hacker News 报告。")

        try:
            if self.config.is_ollama:
                response = ollama.chat(
                    model=self.config.model_name,
                    messages=messages
                )

                # 获取回复内容
                raw_content = response["message"]["content"]

                # 使用正则表达式去掉 <think> 及其内容
                clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
                return clean_content
            else:
                # 调用OpenAI GPT模型生成报告
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # 指定使用的模型版本
                    messages=messages
                )
                LOG.debug("GPT response: {}", response)
                # 返回模型生成的内容
                return response.choices[0].message.content
        except Exception as e:
            # 如果在请求过程中出现异常，记录错误并抛出
            LOG.error(f"生成报告时发生错误：{e}")
            raise

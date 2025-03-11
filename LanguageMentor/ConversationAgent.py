from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from utils.logger import LOG  # 导入日志工具

store={}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    获取指定会话ID的聊天历史。如果该会话ID不存在，则创建一个新的聊天历史实例。
    :param session_id: 会话唯一标识符
    :return:  对应会话的聊天历史对象
    """
    if session_id not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]

class ConversationAgent:
    """
    对话代理类，负责处理与用户的对话。
    """
    def __init__(self):
        self.name = "Conversation Agent"

        with open("prompts/conversation_prompt.txt","r",encoding="utf-8") as file:
            self.system_prompt=file.read().strip()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.chatbot = self.prompt | ChatOllama(
            model="llama3.1",  # 使用的模型名称
            max_tokens=8192,  # 最大生成的token数
            temperature=0.8,  # 生成文本的随机性
        )
        self.chatbot_with_history = RunnableWithMessageHistory(self.chatbot,get_session_history)

        # 配置字典，包含会话ID等可配置参数
        self.config = {"configurable": {"session_id": "abc123"}}

    def chat(self, user_input):
        """
        处理用户输入并生成回复。

        参数:
            user_input (str): 用户输入的消息

        返回:
            str: 代理生成的回复内容
        """
        response = self.chatbot.invoke(
            [HumanMessage(content=user_input)],  # 将用户输入封装为 HumanMessage
        )
        return response.content  # 返回生成的回复内容

    def chat_with_history(self, user_input):
        """
        处理用户输入并生成包含聊天历史的回复，同时记录日志。

        参数:
            user_input (str): 用户输入的消息

        返回:
            str: 代理生成的回复内容
        """
        response = self.chatbot_with_history.invoke(
            [HumanMessage(content=user_input)],  # 将用户输入封装为 HumanMessage
            self.config,  # 传入配置，包括会话ID
        )
        LOG.debug(response)  # 记录调试日志
        return response.content  # 返回生成的回复内容

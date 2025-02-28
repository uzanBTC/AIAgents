# ChatBot.py
from typing import Annotated, Literal
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from Config import config
from BasicToolNode import BasicToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list, add_messages]

class ChatBot:
    def __init__(self):
        self.model_type = config.get("MODEL_TYPE", "ollama")
        self.model_name = config.get("MODEL_NAME", "deepseek-r1:7b")
        self.tools_enabled = config.get("TOOLS_ENABLED", False)

        # 初始化工具
        self.tools = [TavilySearchResults(max_results=2)] if self.tools_enabled else []
        self.tool_node = BasicToolNode(tools=self.tools) if self.tools_enabled else None

        # 初始化模型
        self.llm_model = ChatOllama(model=self.model_name, streaming=True)
        self.chat_model = self._configure_model()

        # 构建状态图
        self.graph = self._build_graph()

    def _configure_model(self):
        """根据模型类型配置 chat_model"""
        if not self.tools_enabled:
            return self.llm_model  # Ollama 不需要绑定工具
        return self.llm_model.bind_tools(self.tools)  # 其他模型绑定工具

    def _chatbot(self, state: State):
        """聊天机器人节点逻辑"""
        return {"messages": [self.chat_model.invoke(state["messages"])]}

    def _route_tools(self, state: State) -> Literal["tools", "__end__"]:
        """路由函数，检查是否有工具调用"""
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"输入状态中未找到消息: {state}")

        if self.tools_enabled and hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "__end__"

    def _build_graph(self):
        """构建状态图"""
        graph_builder = StateGraph(State)

        # 添加 chatbot 节点
        graph_builder.add_node("chatbot", self._chatbot)

        memory = MemorySaver()

        # 如果启用了工具，添加工具节点和条件边
        if self.tools_enabled and self.tool_node:
            graph_builder.add_node("tools", self.tool_node)
            graph_builder.add_conditional_edges(
                "chatbot",
                self._route_tools,
                {"tools": "tools", "__end__": "__end__"}
            )
            graph_builder.add_edge("tools", "chatbot")
        else:
            graph_builder.add_edge("chatbot", END)

        # 设置起始边
        graph_builder.add_edge(START, "chatbot")

        return graph_builder.compile(checkpointer=memory)

    def run(self):
        config = {"configurable": {"thread_id": "1"}}
        """启动聊天机器人"""
        # 进入一个无限循环，用于模拟持续的对话
        while True:
            # 获取用户输入
            user_input = input("User: ")

            # 如果用户输入 "quit"、"exit" 或 "q"，则退出循环，结束对话
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")  # 打印告别语
                break  # 退出循环

            # 使用 graph.stream 处理用户输入，并生成机器人的回复
            # "messages" 列表中包含用户的输入，传递给对话系统
            events=self.graph.stream(
                {"messages": [("user", user_input)]},  # 第一个参数传入用户的输入消息，消息格式为 ("user", "输入内容")
                config,  # 第二个参数用于指定线程配置，包含线程 ID
                stream_mode="values"  # stream_mode 设置为 "values"，表示返回流式数据的值
            )

            for event in events:
                event["messages"][-1].pretty_print()


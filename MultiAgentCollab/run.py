import functools
import operator
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool  # 导入 tool 装饰器

# 加载环境变量
load_dotenv(dotenv_path="3pp_config.env")


# 初始化工具
def initialize_tools():
    """初始化 Tavily 搜索工具和 Python REPL 工具"""
    tavily_tool = TavilySearchResults(max_results=5)

    return tavily_tool


# 定义 python_repl 工具并添加 @tool 装饰器
@tool
def python_repl(
        code: Annotated[str, "要执行的 Python 代码，用于生成图表"],
):
    """执行 Python 代码。如果想查看某个值的输出，需要使用 `print(...)` 打印出来，用户可以看到结果。"""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        return f"执行失败。错误: {repr(e)}"
    result_str = f"成功执行:\n```python\n{code}\n```\n输出: {result}"
    return result_str + "\n\n如果所有任务已完成，请回复 FINAL ANSWER。"


# 创建代理
def create_agent(llm, tools, system_message: str):
    """根据给定的 LLM、工具和系统消息创建代理"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个乐于助人的 AI 助手，与其他助手协作。"
                " 使用提供的工具逐步回答问题。"
                " 如果你无法完全回答，没关系，其他助手会接手继续完成。"
                " 执行你能做的部分以推进任务。"
                " 如果你或其他助手得到了最终答案或交付物，"
                " 请在回复前加上 FINAL ANSWER，以便团队知道可以结束。"
                " 你可以使用以下工具: {tool_names}。\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


# 代理节点函数
def agent_node(state, agent, name):
    """处理代理的调用并更新状态"""
    result = agent.invoke(state)
    if not isinstance(result, ToolMessage):
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name
    }


# 创建代理节点
def create_agent_node(agent, name):
    """创建指定名称的代理节点"""
    return functools.partial(agent_node, agent=agent, name=name)


# 路由函数
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    """根据状态决定下一步动作"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    return "continue"


# 定义状态类
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# 构建工作流
def build_workflow(research_agent, chart_agent, tools):
    """构建并编译状态图工作流"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("Researcher", create_agent_node(research_agent, "Researcher"))
    workflow.add_node("Chart_Generator", create_agent_node(chart_agent, "Chart_Generator"))
    workflow.add_node("call_tool", ToolNode(tools))

    # 添加条件边
    workflow.add_conditional_edges(
        "Researcher",
        router,
        {"continue": "Chart_Generator", "call_tool": "call_tool", "__end__": END}
    )
    workflow.add_conditional_edges(
        "Chart_Generator",
        router,
        {"continue": "Researcher", "call_tool": "call_tool", "__end__": END}
    )
    workflow.add_conditional_edges(
        "call_tool",
        lambda x: x["sender"],
        {"Researcher": "Researcher", "Chart_Generator": "Chart_Generator"}
    )

    # 设置入口点
    workflow.add_edge(START, "Researcher")

    return workflow.compile()


# 程序入口函数
def run():
    """启动多代理工作流"""
    try:
        # 初始化工具
        tavily_tool = initialize_tools()
        # 使用装饰后的 python_repl 工具
        tools = [tavily_tool, python_repl]

        # 初始化语言模型
        research_llm = ChatOllama(model="llama3.1")
        chart_llm = ChatOllama(model="llama3.1")

        # 创建代理
        research_agent = create_agent(
            research_llm,
            [tavily_tool],
            "在使用搜索引擎之前，仔细思考并澄清查询内容。"
            "然后，一次性搜索解决查询的所有方面。"
        )
        chart_agent = create_agent(
            chart_llm,
            [python_repl],  # 使用装饰后的 python_repl 工具
            "根据提供的数据创建清晰且用户友好的图表，并能够保存本地，以便用户查看。"
        )

        # 构建工作流
        graph = build_workflow(research_agent, chart_agent, tools)

        # 定义初始任务
        initial_message = HumanMessage(
            content="获取美国 2000 年至 2020 年的 GDP 数据，"
                    "然后用 Python 绘制折线图。生成图表后，保存到本地，然后结束任务。"
        )

        # 执行工作流
        events = graph.stream(
            {"messages": [initial_message]},
            {"recursion_limit": 20},
            stream_mode="values"
        )

        # 输出结果
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()

    except Exception as e:
        print(f"执行过程中出错: {e}")


# 主程序入口
if __name__ == "__main__":
    run()

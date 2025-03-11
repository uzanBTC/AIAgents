import argparse
from functools import partial
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama
from IPython.display import Markdown, display

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

MAX_ROUND = 6

# ------------------------------
# 装饰器和辅助打印函数
# ------------------------------
def track_steps(func):
    step_counter = {'count': 0}
    def wrapper(event, *args, **kwargs):
        step_counter['count'] += 1
        display(Markdown(f"## Round {step_counter['count']}"))
        return func(event, *args, **kwargs)
    return wrapper

@track_steps
def pretty_print_event_markdown(event):
    if 'writer' in event:
        generate_md = "#### 写作生成:\n"
        for message in event['writer']['messages']:
            generate_md += f"- {message.content}\n"
        display(Markdown(generate_md))
    if 'reflect' in event:
        reflect_md = "#### 评论反思:\n"
        for message in event['reflect']['messages']:
            reflect_md += f"- {message.content}\n"
        display(Markdown(reflect_md))

# ------------------------------
# 用户输入辅助函数
# ------------------------------
def get_user_input(prompt_message: str, default: str = None) -> str:
    prompt_text = f"{prompt_message} (默认: {default}): " if default else f"{prompt_message}: "
    user_input = input(prompt_text)
    return user_input.strip() or default

# ------------------------------
# 创建 writer 和 reflect 实例
# ------------------------------
def create_chat_instances():
    writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a writing assistant tasked with creating well-crafted, coherent, and engaging articles based on the user's request. "
                "Focus on clarity, structure, and quality."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a teacher grading an article submission. Provide critique and recommendations for the user's submission."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    writer = writer_prompt | ChatOllama(
        model="llama3.1",
        max_tokens=8192,
        temperature=1.2,
    )
    reflect = reflection_prompt | ChatOllama(
        model="llama3.1",
        max_tokens=8192,
        temperature=0.2,
    )
    return writer, reflect

# ------------------------------
# 生成文章和反思（同步调用）
# ------------------------------
def generate_article(topic: str, writer) -> str:
    article = ""
    topic_message = HumanMessage(content=topic)
    print("\n生成文章中...\n")
    for chunk in writer.stream({"messages": [topic_message]}):
        print(chunk.content, end="")
        article += chunk.content
    return article

def generate_reflection(topic: str, article: str, reflect) -> str:
    reflection = ""
    print("\n生成反思中...\n")
    for chunk in reflect.stream({"messages": [HumanMessage(content=topic), HumanMessage(content=article)]}):
        print(chunk.content, end="")
        reflection += chunk.content
    return reflection

# ------------------------------
# 同步状态图节点函数
# ------------------------------
def generation_node(state: dict, writer) -> dict:
    result = ""
    for chunk in writer.stream({"messages": state["messages"]}):
        result += chunk.content
    return {"messages": [HumanMessage(content=result)]}

def reflection_node(state: dict, reflect) -> dict:
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    result = ""
    for chunk in reflect.stream({"messages": translated}):
        result += chunk.content
    return {"messages": [HumanMessage(content=result)]}

def build_and_run_state_graph(topic: str, writer, reflect):
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # 将节点函数与 writer/reflect 绑定
    gen_node = lambda state: generation_node(state, writer)
    ref_node = lambda state: reflection_node(state, reflect)

    def should_continue(state: State):
        if len(state["messages"]) > MAX_ROUND:
            return END  # 达到轮次后结束
        return "reflect"  # 否则进入反思节点

    builder = StateGraph(State)
    builder.add_node("writer", gen_node)
    builder.add_node("reflect", ref_node)
    builder.add_edge(START, "writer")
    builder.add_conditional_edges("writer", should_continue)
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    inputs = {"messages": [HumanMessage(content=topic)]}
    config = {"configurable": {"thread_id": "1"}}
    print("\n开始状态图流程...\n")
    # 使用同步调用 invoke 执行状态图
    result = graph.invoke(inputs, config)
    pretty_print_event_markdown(result)
    return result

def main():
    parser = argparse.ArgumentParser(description="文章生成与反思流程工具")
    parser.add_argument(
        "--topic",
        type=str,
        default="参考水浒传的风格，改写吴承恩的西游记中任意篇章",
        help="文章生成的主题描述"
    )
    parser.add_argument(
        "--graph_topic",
        type=str,
        default="参考西游记唐僧的说话风格，写一篇奉劝年轻人努力工作的文章",
        help="状态图流程的初始主题描述"
    )
    args = parser.parse_args()

    topic = args.topic or get_user_input("请输入写作生成主题", "参考水浒传的风格，改写吴承恩的西游记中任意篇章")
    graph_topic = args.graph_topic or get_user_input("请输入状态图生成主题", "参考西游记唐僧的说话风格，写一篇奉劝年轻人努力工作的文章")

    writer, reflect = create_chat_instances()

    article = generate_article(topic, writer)
    print("\n\n文章生成完毕。\n")
    reflection = generate_reflection(topic, article, reflect)
    print("\n\n反思生成完毕。\n")

    # 为状态图流程重新创建新的实例，确保独立使用
    writer_graph, reflect_graph = create_chat_instances()
    build_and_run_state_graph(graph_topic, writer_graph, reflect_graph)

if __name__ == "__main__":
    main()

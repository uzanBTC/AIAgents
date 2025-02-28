import gradio as gr  # 导入gradio库用于创建GUI

from config import Config  # 导入配置管理模块
from github_client import GitHubClient  # 导入用于GitHub API操作的客户端
from report_generator import ReportGenerator  # 导入报告生成器模块
from llm import LLM  # 导入可能用于处理语言模型的LLM类
from src.hackernews_client import HackerNewsClient
from subscription_manager import SubscriptionManager  # 导入订阅管理器
from logger import LOG  # 导入日志记录器

# 创建各个组件的实例
config = Config()
github_client = GitHubClient(config.github_token)
llm = LLM()
report_generator = ReportGenerator(llm)
subscription_manager = SubscriptionManager(config.subscriptions_file)
hackernewsClient=HackerNewsClient()

def export_progress_by_date_range(repo, days):
    # 定义一个函数，用于导出和生成指定时间范围内项目的进展报告
    raw_file_path = github_client.export_progress_by_date_range(repo, days)  # 导出原始数据文件路径
    report, report_file_path = report_generator.generate_report_by_date_range(raw_file_path, days)  # 生成并获取报告内容及文件路径

    return report, report_file_path  # 返回报告内容和报告文件路径

def export_hackernews_trends():
    raw_file_path=hackernewsClient.export_hackernews_top_stories()
    report, report_file_path = report_generator.generate_hackernews_trends_report(raw_file_path)
    return report, report_file_path  # 返回报告内容和报告文件路径

# 创建Gradio界面
# demo = gr.Interface(
#     fn=export_progress_by_date_range,  # 指定界面调用的函数
#     title="GitHubSentinel",  # 设置界面标题
#     inputs=[
#         gr.Dropdown(
#             subscription_manager.list_subscriptions(), label="订阅列表", info="已订阅GitHub项目"
#         ),  # 下拉菜单选择订阅的GitHub项目
#         gr.Slider(value=2, minimum=1, maximum=7, step=1, label="报告周期", info="生成项目过去一段时间进展，单位：天"),
#         # 滑动条选择报告的时间范围
#     ],
#     outputs=[gr.Markdown(), gr.File(label="下载报告")],  # 输出格式：Markdown文本和文件下载
# )

with gr.Blocks() as demo:
    gr.Markdown("# GitHub Sentinel & HackerNews Trends")  # 页面标题

    with gr.Tab("GitHub 项目进展"):
        gr.Markdown("### 选择GitHub项目和报告周期")  # 说明
        repo_dropdown = gr.Dropdown(
            subscription_manager.list_subscriptions(), label="订阅列表", info="已订阅GitHub项目"
        )  # 下拉菜单选择订阅的GitHub项目
        days_slider = gr.Slider(value=2, minimum=1, maximum=7, step=1, label="报告周期", info="生成项目过去一段时间进展，单位：天")
        progress_report_output = gr.Markdown()  # 用于显示报告内容
        progress_report_file = gr.File(label="下载报告")  # 用于下载报告文件

        # 按钮点击事件触发函数
        generate_progress_button = gr.Button("生成GitHub项目进展报告")
        generate_progress_button.click(
            fn=export_progress_by_date_range,
            inputs=[repo_dropdown, days_slider],
            outputs=[progress_report_output, progress_report_file]
        )

    with gr.Tab("HackerNews Trends"):
        gr.Markdown("### 获取最新的HackerNews热门故事趋势")  # 说明
        hackernews_report_output = gr.Markdown()  # 用于显示报告内容
        hackernews_report_file = gr.File(label="下载HackerNews报告")  # 用于下载报告文件

        # 按钮点击事件触发函数
        generate_hackernews_button = gr.Button("生成HackerNews趋势报告")
        generate_hackernews_button.click(
            fn=export_hackernews_trends,
            inputs=[],
            outputs=[hackernews_report_output, hackernews_report_file]
        )

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)  # 启动界面并设置为公共可访问
    # 可选带有用户认证的启动方式
    # demo.launch(share=True, server_name="0.0.0.0", auth=("django", "1234"))
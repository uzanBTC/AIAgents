import gradio as gr
from rag import RAGSystem
from utils import setup_logger

logger = setup_logger()

# 初始化 RAG 系统
rag_system = RAGSystem()

# Gradio 前端模块
def add_document_interface(document):
    if not document.strip():
        return "文档内容不能为空。"
    if rag_system.is_crypto_related(document):
        rag_system.add_document(document)
        return "文档已成功添加到向量数据库。"
    else:
        return "文档不属于数字货币相关内容，未添加。"

def retrieve_interface(query):
    if not query.strip():
        return "查询内容不能为空。"
    return rag_system.generate_answer(query)

if __name__ == "__main__":
    logger.info("启动 Crypto RAG 项目")
    with gr.Blocks(title="Crypto RAG System") as app:
        with gr.Tab("Add Documents"):
            document_input = gr.Textbox(label="Document", lines=5, placeholder="请输入文档内容")
            add_button = gr.Button("Add Document")
            output = gr.Textbox(label="Result")
            add_button.click(add_document_interface, inputs=document_input, outputs=output)

        with gr.Tab("Retrieve"):
            query_input = gr.Textbox(label="Query", lines=2, placeholder="请输入您的问题")
            retrieve_button = gr.Button("Retrieve")
            answer_output = gr.Textbox(label="Answer")
            retrieve_button.click(retrieve_interface, inputs=query_input, outputs=answer_output)

    app.launch(server_name="localhost", server_port=7860)

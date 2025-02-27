from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import HumanMessage
from langchain_community.vectorstores import PGVector
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from Config import config
from utils import setup_logger, format_docs, clean_think_tags

class RAGSystem:
    """RAG 系统类，封装文档验证、添加和回答生成功能"""

    def __init__(self):
        """初始化 RAG 系统"""
        self.logger = setup_logger()
        self.llm = ChatOllama(model=config.MODEL, temperature=0.6, max_tokens=10)
        self.embeddings = OllamaEmbeddings(model=config.MODEL)
        self.connection_string = config.CONNECTION_STRING

        # 定义 Prompt 模板
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""基于以下内容回答问题：
                
                {context}
                
                问题：{question}
                """
        )

        # 初始化向量存储
        self.vector_store = PGVector(
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            collection_name="documents"
        )

        self.logger.info("RAG 系统初始化完成")

    def is_crypto_related(self, document):
        """判断文档是否与数字货币相关"""
        try:
            prompt = f"请判断以下文档是否与数字货币相关。只回答'是'或'否'。\n\n{document}"
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw_answer = response.content.strip().lower()
            clean_answer = clean_think_tags(raw_answer)
            self.logger.info(f"文档验证: 原始输出={raw_answer}, 清洗后={clean_answer}")
            return '是' in clean_answer
        except Exception as e:
            self.logger.error(f"验证文档时出错: {e}")
            return False

    def add_document(self, document):
        """将文档添加到向量数据库"""
        try:
            doc = Document(page_content=document)
            self.vector_store.add_documents([doc])
            self.logger.info(f"成功添加文档: {document[:50]}...")
        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")

    def generate_answer(self, question):
        """生成回答"""
        try:
            if not question or not isinstance(question, str):
                raise ValueError("问题必须是非空的字符串")

            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            answer = rag_chain.invoke(question)
            answer= clean_think_tags(answer)
            self.logger.info(f"生成回答: 问题={question}, 回答={answer}")
            return answer
        except Exception as e:
            self.logger.error(f"生成回答失败: {e}")
            return "抱歉，无法生成回答，请稍后重试。"

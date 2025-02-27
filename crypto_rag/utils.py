import logging
import re
from langchain.docstore.document import Document

# 配置日志
logging.basicConfig(
    filename='crypto_rag.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logger():
    return logger

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def clean_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().lower()

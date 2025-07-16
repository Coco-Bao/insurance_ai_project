import os
import json
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from huggingface_hub import snapshot_download

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "你的OpenAI_API_Key")
CHROMA_DB_PATH = "data/chroma_db"

class ProductDatabaseBuilder:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

        print("下載 RapidOCR 模型（首次運行會自動快取）...")
        model_cache_path = snapshot_download(repo_id="SWHL/RapidOCR")

        self.ocr_options = RapidOcrOptions(
            det_model_path=os.path.join(model_cache_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"),
            rec_model_path=os.path.join(model_cache_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"),
            cls_model_path=os.path.join(model_cache_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")
        )

        self.pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=self.ocr_options,
            do_table_structure=True,
        )

    def _split_docs_by_titles(self, doc_json: Dict[str, Any]) -> List[Document]:
        """
        將Docling導出的結構化JSON依據標題切分成多個段落。
        假設每個標題是一個區塊的起始。此處示範以 'heading' 或 'title' 字段做標題判斷。
        """
        blocks = doc_json.get("blocks", [])
        chunks = []
        current_chunk = []
        current_title = None

        def flush_current_chunk():
            if current_chunk:
                text = "\n".join(current_chunk).strip()
                if text:
                    meta = {"title": current_title}
                    chunks.append(Document(page_content=text, metadata=meta))

        for block in blocks:
            # 取得本區塊標題 (示意，有些文件可能用不同欄位，請按實際調整)
            title = block.get("heading") or block.get("title") or None
            text = block.get("text") or ""

            if title:
                # 有新標題，先flush前一段
                flush_current_chunk()
                current_title = title
                current_chunk = [text]
            else:
                # 無新標題，合併進目前chunk
                if text:
                    current_chunk.append(text)

        # 最後一段
        flush_current_chunk()

        # 對每個大段落用 LangChain文本拆分器進一步細分
        refined_chunks = []
        for doc in chunks:
            split_subdocs = self.text_splitter.split_documents([doc])
            refined_chunks.extend(split_subdocs)

        return refined_chunks

    def load_and_split_pdf(self, pdf_path: str) -> List[Document]:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        print(f"開始用 Docling + RapidOCR 處理 PDF：{pdf_path}")
        conversion_result = converter.convert(pdf_path)
        doc = conversion_result.document

        # 導出全文結構化JSON
        doc_json_str = doc.export_to_json(indent=2, ensure_ascii=False)
        doc_json = json.loads(doc_json_str)
        print("成功導出結構化JSON")

        # 以標題為單位切分
        split_docs = self._split_docs_by_titles(doc_json)
        print(f"經過標題拆分和文本細分，產生 {len(split_docs)} 個段落")

        return split_docs

    def build_product_database(self, pdf_path: str):
        split_docs = self.load_and_split_pdf(pdf_path)

        documents = []
        for idx, doc in enumerate(split_docs):
            metadata = {
                "source_file": os.path.basename(pdf_path),
                "chunk_index": idx,
                "title": doc.metadata.get("title", "")
            }
            documents.append(Document(page_content=doc.page_content, metadata=metadata))

        vectordb = Chroma.from_documents(documents, self.embeddings, persist_directory=CHROMA_DB_PATH)
        vectordb.persist()
        print("產品資料庫建立完成並持久化。")

class InsuranceAIAgent:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=self.embeddings)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})

    def process_customer_input(self, input_text: str) -> Dict:
        related_docs = self.retriever.get_relevant_documents(input_text)
        context_text = "\n".join(doc.page_content for doc in related_docs)

        prompt_template = (
            "你是一位保險顧問，根據以下產品資料與客戶需求推薦最適合的保險產品，"
            "並指出推薦依據是資料庫中哪些文件和章節。\n\n"
            "客戶需求：\n{customer_input}\n\n"
            "相關產品資料（結構化JSON片段）：\n{context}\n\n"
            "請給出詳細推薦理由和支持依據。"
        )
        prompt = prompt_template.format(customer_input=input_text, context=context_text)
        response = self.llm(prompt)

        return {
            "recommendation": response,
            "supported_by": [
                {
                    "source_file": doc.metadata.get("source_file"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "content_preview": doc.page_content[:300]
                }
                for doc in related_docs
            ]
        }

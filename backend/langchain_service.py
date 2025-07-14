import os
import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import cv2
import numpy as np

CHROMA_DB_PATH = "data/chroma_db"

# 初始化 PaddleOCR，支援繁體中文和英文，GPU設為False（如有GPU可設True）
paddle_ocr = PaddleOCR(lang='ch', use_angle_cls=False)

# 自动加载当前目录下的 .env 文件
load_dotenv()

class PaddleOCRPDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.ocr = paddle_ocr

    def load(self) -> List[Document]:
        # 將PDF每頁轉換為圖片
        pages = convert_from_path(self.pdf_path)
        documents = []
        for i, page in enumerate(pages):
            # 將PIL圖片轉成OpenCV格式（BGR）
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            # 使用PaddleOCR進行文字辨識
            result = self.ocr.ocr(img)
            # 將辨識結果中每行文字取出並合併成一整頁文字
            page_text = "\n".join([line[1][0] for line in result[0]])
            # 封裝成Document物件，並加入頁碼metadata
            documents.append(Document(page_content=page_text, metadata={"page": i + 1}))
        print(f"PaddleOCR共識別 {len(documents)} 頁")
        return documents


class ProductDatabaseBuilder:
    def __init__(self):
        # 初始化OpenAI嵌入模型
        self.embeddings = OpenAIEmbeddings()
        # 初始化ChatOpenAI模型，使用Gemini 2.0 Flash版本
        self.llm = ChatOpenAI(
            model="gemini-2.0-flash",
            temperature=0,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"],
        )
        # 設定文本拆分器，設定塊大小與重疊字元數
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)

    def load_and_split_pdf(self, pdf_path: str) -> List[Document]:
        # 使用PaddleOCRPDFLoader讀取PDF並OCR轉文字
        loader = PaddleOCRPDFLoader(pdf_path)
        raw_docs = loader.load()
        print(f"共讀取 {len(raw_docs)} 個文件塊（頁面）")

        # 對讀取的文件塊進行拆分成更小段落
        split_docs = self.text_splitter.split_documents(raw_docs)
        print(f"拆分成 {len(split_docs)} 個小段落")
        return split_docs

    def structure_document(self, text: str) -> Dict:
        # 定義提示模板，讓LLM結構化非結構化文本
        prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "你是一個保險產品資料結構化專家，"
                "請從以下非結構化文本中提取所有與保險產品相關的重要信息。"
                "這些信息可能包括產品名稱、保障範圍、保費、條款摘要、適用人群、投保條件、理賠流程、責任免除、附加條款、優惠政策、申請方式等，"
                "也可能包含其他任何有價值的細節。"
                "請將提取到的資訊以結構化的JSON格式返回，字段名稱和結構請根據內容靈活設計，盡量全面且清晰。\n\n"
                "文本：\n{text}\n\n"
                "請只返回JSON，不要多餘文字。"
            )
        )
        # 建立LLMChain執行結構化任務
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(text=text)
        print(f"LLM結構化返回結果示例（前500字）：{result[:500]}")

        # 嘗試解析JSON，若失敗則返回原始文本
        try:
            structured_data = json.loads(result)
        except Exception as e:
            print(f"JSON 解析錯誤，返回原始文本: {e}")
            structured_data = {"raw_text": text}
        return structured_data

    def aggregate_structured_jsons(self, json_list: List[Dict]) -> Dict:
        # 將多個JSON片段合併為一個完整結構
        fragments_text = "\n".join(
            [f"片段{i+1}：{json.dumps(j, ensure_ascii=False)}" for i, j in enumerate(json_list)]
        )

        prompt_template = (
            "你是一位保險產品資料結構化專家，"
            "現在我給你多個保險產品資料的JSON片段，這些片段是從同一份文件不同部分提取的。"
            "請你將它們合併成一個完整且無重複的保險產品結構化資料，"
            "補充缺失的信息，並以JSON格式返回。\n\n"
            "{fragments_text}\n\n"
            "請只返回合併後的JSON，不要多餘文字。"
        )

        prompt = PromptTemplate(input_variables=["fragments_text"], template=prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(fragments_text=fragments_text)

        print(f"聚合結果示例（前500字）：{response[:500]}")

        try:
            structured_data = json.loads(response)
        except Exception as e:
            print(f"聚合JSON解析錯誤: {e}")
            structured_data = {}
        return structured_data

    def build_product_database(self, pdf_path: str):
        # 讀取並拆分PDF文件
        split_docs = self.load_and_split_pdf(pdf_path)

        structured_jsons = []
        # 對拆分後的每個段落進行結構化處理
        for idx, doc in enumerate(split_docs):
            print(f"結構化處理第 {idx + 1}/{len(split_docs)} 段")
            structured_json = self.structure_document(doc.page_content)
            structured_jsons.append(structured_json)

        print("開始聚合所有結構化結果...")
        final_structured = self.aggregate_structured_jsons(structured_jsons)
        print("聚合完成，結果示例:", json.dumps(final_structured, ensure_ascii=False)[:500])

        # 將聚合後的結構化資料封裝成Document
        structured_documents = [
            Document(
                page_content=json.dumps(final_structured, ensure_ascii=False),
                metadata={
                    "source_file": os.path.basename(pdf_path),
                    "chunk_index": "aggregated",
                    "structured_data": final_structured
                }
            )
        ]

        # 建立並持久化向量資料庫
        vectordb = Chroma.from_documents(structured_documents, self.embeddings, persist_directory=CHROMA_DB_PATH)
        vectordb.persist()
        print("產品資料庫建立完成並持久化。")


class InsuranceAIAgent:
    def __init__(self):
        # 初始化OpenAI嵌入模型與LLM
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model="gemini-2.0-flash",
            temperature=0,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"],
        )
        # 設定文本拆分器
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        # 載入持久化的向量資料庫
        self.vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=self.embeddings)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})

    def extract_customer_profile(self, pdf_path: str) -> str:
        print(f"解析客戶PDF檔案: {pdf_path}")
        # 使用PaddleOCRPDFLoader讀取客戶PDF並OCR轉文字
        loader = PaddleOCRPDFLoader(pdf_path)
        raw_docs = loader.load()
        # 對文字進行拆分
        split_docs = self.text_splitter.split_documents(raw_docs)

        # 合併拆分後文字為完整客戶文本
        full_customer_text = "\n".join([doc.page_content for doc in split_docs])

        # 定義提示模板，讓LLM提取客戶畫像
        prompt = PromptTemplate(
            input_variables=["customer_full_text"],
            template=(
                "你是一個客戶資料分析專家，請從以下客戶提供的非結構化文件中，"
                "提取所有重要的客戶資訊，例如年齡、職業、家庭狀況、健康狀況、"
                "已有的保險、財務狀況、主要關注點、潛在需求、偏好等。\n"
                "請將這些資訊整理成一段簡潔、全面的客戶畫像描述，以便於保險產品推薦。\n\n"
                "客戶文件內容：\n{customer_full_text}\n\n"
                "客戶畫像描述："
            )
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        customer_profile_text = chain.run(customer_full_text=full_customer_text)
        print(f"提取的客戶畫像: {customer_profile_text[:200]}...")
        return customer_profile_text

    def recommend_products(self, customer_profile_text: str) -> Dict:
        # 根據客戶畫像從向量資料庫檢索相關產品資料
        related_docs = self.retriever.get_relevant_documents(customer_profile_text)
        context_text = "\n".join([doc.page_content for doc in related_docs])

        # 定義推薦提示模板
        prompt_template = (
            "你是一個專業的保險顧問，根據以下客戶畫像和相關產品資料，"
            "請推薦最適合該客戶的保險產品。\n\n"
            "推薦時，請明確指出推薦的產品名稱，並詳細說明理由，"
            "以及支持該推薦的產品特點與客戶需求的匹配點。\n"
            "最後，請列出支持推薦的依據，包括原產品文件的名稱、所引用的段落索引，以及引用的產品資料內容預覽。\n\n"
            "客戶畫像：\n{customer_profile}\n\n"
            "相關產品資料（JSON格式）：\n{context}\n\n"
            "請給出詳細推薦理由和支持依據。"
        )
        prompt = prompt_template.format(customer_profile=customer_profile_text, context=context_text)
        response = self.llm.invoke(prompt).content

        return {
            "recommendation": response,
            "supported_by": [
                {
                    "source_file": doc.metadata.get("source_file", "N/A"),
                    "chunk_index": doc.metadata.get("chunk_index", "N/A"),
                    "structured_data_preview": doc.page_content[:300] + "..."
                } for doc in related_docs
            ]
        }

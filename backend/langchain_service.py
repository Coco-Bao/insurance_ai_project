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

import easyocr
from pdf2image import convert_from_path
import cv2
import numpy as np

CHROMA_DB_PATH = "data/chroma_db"

# 設置 Gemini API Key 和 Base URL
os.environ["OPENAI_API_KEY"] = "AIzaSyCpoWTEbR9ggzR_74bkUUdjir_2Kw8ALm0"
os.environ["OPENAI_API_BASE"] = "https://generativelanguage.googleapis.com/v1beta/openai"

# 初始化 EasyOCR，支持繁体中文和英文
easyocr_reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)  # gpu=True 如果你有GPU支持

class EasyOCRPDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def load(self) -> List[Document]:
        pages = convert_from_path(self.pdf_path)
        documents = []
        for i, page in enumerate(pages):
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            result = easyocr_reader.readtext(img, detail=0)  # detail=0 只返回文本
            page_text = "\n".join(result)
            documents.append(Document(page_content=page_text, metadata={"page": i + 1}))
        print(f"EasyOCR共识别 {len(documents)} 页")
        return documents


class ProductDatabaseBuilder:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model="gemini-2.0-flash",
            temperature=0,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"],
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)

    def load_and_split_pdf(self, pdf_path: str) -> List[Document]:
        loader = EasyOCRPDFLoader(pdf_path)
        raw_docs = loader.load()
        print(f"共讀取 {len(raw_docs)} 個文件塊（頁面）")

        split_docs = self.text_splitter.split_documents(raw_docs)
        print(f"拆分成 {len(split_docs)} 個小段落")
        return split_docs

    def structure_document(self, text: str) -> Dict:
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
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(text=text)
        print(f"LLM結構化返回結果示例（前500字）：{result[:500]}")

        try:
            structured_data = json.loads(result)
        except Exception as e:
            print(f"JSON 解析錯誤，返回原始文本: {e}")
            structured_data = {"raw_text": text}
        return structured_data

    def aggregate_structured_jsons(self, json_list: List[Dict]) -> Dict:
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
        split_docs = self.load_and_split_pdf(pdf_path)

        structured_jsons = []
        for idx, doc in enumerate(split_docs):
            print(f"結構化處理第 {idx + 1}/{len(split_docs)} 段")
            structured_json = self.structure_document(doc.page_content)
            structured_jsons.append(structured_json)

        print("開始聚合所有結構化結果...")
        final_structured = self.aggregate_structured_jsons(structured_jsons)
        print("聚合完成，結果示例:", json.dumps(final_structured, ensure_ascii=False)[:500])

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

        vectordb = Chroma.from_documents(structured_documents, self.embeddings, persist_directory=CHROMA_DB_PATH)
        vectordb.persist()
        print("產品資料庫建立完成並持久化。")


class InsuranceAIAgent:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model="gemini-2.5-flash",
            temperature=0,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"],
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        self.vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=self.embeddings)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})

    def extract_customer_profile(self, pdf_path: str) -> str:
        print(f"解析客戶PDF檔案: {pdf_path}")
        loader = EasyOCRPDFLoader(pdf_path)
        raw_docs = loader.load()
        split_docs = self.text_splitter.split_documents(raw_docs)

        full_customer_text = "\n".join([doc.page_content for doc in split_docs])

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
        related_docs = self.retriever.get_relevant_documents(customer_profile_text)
        context_text = "\n".join([doc.page_content for doc in related_docs])

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

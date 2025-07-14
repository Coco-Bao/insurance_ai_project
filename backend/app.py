from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid

# 請確保這裡匯入的是你修改後的OCR類別和業務邏輯類別
from langchain_service import ProductDatabaseBuilder, InsuranceAIAgent, PaddleOCRPDFLoader
from pillow_heif import register_heif_opener

register_heif_opener()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'backend/uploaded_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db_builder = ProductDatabaseBuilder()
ai_agent = InsuranceAIAgent()


@app.route('/upload_product_pdf', methods=['POST'])
def upload_product_pdf():
    # 檢查是否有檔案上傳
    if 'file' not in request.files:
        return jsonify({"error": "請上傳產品PDF檔案"}), 400
    file = request.files['file']
    # 檢查檔案名稱是否為空
    if file.filename == '':
        return jsonify({"error": "檔案名稱不可為空"}), 400

    # 只允許PDF格式
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext != 'pdf':
        return jsonify({"error": "只支援PDF格式"}), 400

    # 產生唯一檔名，避免檔案衝突
    unique_filename = f"product_{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    # 儲存上傳的檔案
    file.save(file_path)
    print(f"File saved to: {file_path}")

    try:
        # 建立產品資料庫
        db_builder.build_product_database(file_path)
        global ai_agent
        # 重新載入AI代理，確保使用最新資料庫
        ai_agent = InsuranceAIAgent()
        return jsonify({"message": "產品資料庫更新成功"})
    except Exception as e:
        print(f"產品PDF處理失敗: {e}")
        return jsonify({"error": f"產品PDF處理失敗: {str(e)}"}), 500
    finally:
        # 處理完成後刪除暫存檔案
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route('/upload_customer_pdf_and_recommend', methods=['POST'])
def upload_customer_pdf_and_recommend():
    # 檢查是否有檔案上傳
    if 'file' not in request.files:
        return jsonify({"error": "請上傳客戶資料PDF檔案"}), 400
    file = request.files['file']
    # 檢查檔案名稱是否為空
    if file.filename == '':
        return jsonify({"error": "檔案名稱不可為空"}), 400

    # 只允許PDF格式
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext != 'pdf':
        return jsonify({"error": "只支援PDF格式"}), 400

    # 產生唯一檔名，避免檔案衝突
    unique_filename = f"customer_{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    # 儲存上傳的檔案
    file.save(file_path)
    print(f"File saved to: {file_path}")

    try:
        # 使用AI代理提取客戶畫像
        customer_profile = ai_agent.extract_customer_profile(file_path)
        # 根據客戶畫像推薦保險產品
        result = ai_agent.recommend_products(customer_profile)
        return jsonify(result)
    except Exception as e:
        print(f"客戶PDF處理失敗: {e}")
        return jsonify({"error": f"客戶PDF處理失敗: {str(e)}"}), 500
    finally:
        # 處理完成後刪除暫存檔案
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == '__main__':
    print("後端服務準備就緒。")
    app.run(host='0.0.0.0', debug=True, port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from langchain_service_Docling import ProductDatabaseBuilder, InsuranceAIAgent
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
        return jsonify({"error": "請上傳產品 PDF 檔案"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "檔案名稱不可為空"}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext != 'pdf':
        return jsonify({"error": "只支援 PDF 格式"}), 400

    # 產生唯一檔名，避免檔案衝突
    unique_filename = f"product_{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    # 儲存上傳的檔案
    file.save(file_path)
    print(f"檔案已儲存至: {file_path}")

    try:
        # 建立產品資料庫
        db_builder.build_product_database(file_path)
        global ai_agent
        # 重新載入 AI 代理，確保使用最新資料庫
        ai_agent = InsuranceAIAgent()
        return jsonify({"message": "產品資料庫更新成功"})
    except Exception as e:
        print(f"產品 PDF 處理失敗: {e}")
        return jsonify({"error": f"產品 PDF 處理失敗: {str(e)}"}), 500
    finally:
        # 處理完成後刪除暫存檔案
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    customer_input = data.get("customer_data")
    if not customer_input:
        return jsonify({"error": "請提供客戶需求資料"}), 400

    try:
        result = ai_agent.process_customer_input(customer_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

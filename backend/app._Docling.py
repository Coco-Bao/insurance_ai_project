from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from langchain_service import ProductDatabaseBuilder, InsuranceAIAgent

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db_builder = ProductDatabaseBuilder()
ai_agent = InsuranceAIAgent()

@app.route("/upload_product_pdf", methods=["POST"])
def upload_product_pdf():
    if "file" not in request.files:
        return jsonify({"error": "請上傳產品PDF文件"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名不可為空"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    if ext != "pdf":
        return jsonify({"error": "只支持PDF格式"}), 400

    unique_filename = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    try:
        db_builder.build_product_database(file_path)
        return jsonify({"message": "產品資料庫更新成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    customer_input = data.get("customer_data")
    if not customer_input:
        return jsonify({"error": "請提供客戶資料"}), 400

    try:
        result = ai_agent.process_customer_input(customer_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

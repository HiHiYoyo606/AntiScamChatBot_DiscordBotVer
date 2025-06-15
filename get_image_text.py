import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def generate_text_from_image_gemini(
    image_bytes: bytes,
    prompt_text: str = "請複述一遍照片中你看到的文字(只輸出照片中的文字，而不是描述整張圖片，如果看起來沒有文字或不明顯則不回傳任何資訊)",
    model_name: str = "gemini-2.0-flash" # Or "gemini-pro-vision" for older model
):
    """
    使用 Gemini API 從圖片中提取文字。

    Args:
        image_bytes: 圖片的原始位元組數據。
        prompt_text: 伴隨圖片的文字提示。
        model_name: 要使用的 Gemini 模型名稱 (需支援視覺)。

    Returns:
        提取到的文字字串(可能為空)，或者在發生錯誤時返回錯誤訊息。
    """
    try:
        # 設定 API 金鑰

        # 準備圖片部分
        image_part = {
            "mime_type": "image/png", # 或 "image/jpeg" 等，根據你的圖片格式
            "data": image_bytes
        }

        # 初始化模型
        model = genai.GenerativeModel(model_name)
        contents = [prompt_text, image_part]
        
        # 非流式獲取 (如果回應通常較短，這樣更簡單)
        response = model.generate_content(contents)

        # 檢查是否有回應被阻擋
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"錯誤：請求被阻擋，原因：{response.prompt_feedback.block_reason.name}"
        
        # 檢查是否有有效的候選內容
        if not response.candidates or not response.candidates[0].content.parts:
             return "錯誤：模型未生成內容或返回空回應。"
        
        # 從非流式回應中獲取文字
        extracted_text = response.text
        return extracted_text.strip()

    except Exception as e:
        # 在實際應用中，你可能想使用 logging 模組來記錄錯誤
        raise Exception(f"get_image_text 發生錯誤: {e}") from e
import base64
import os
from google import genai
from google.genai import types

def generate(image_file,api_key):
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    client = genai.Client(
        api_key=api_key,
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=base64.b64decode(encoded_image),
                ),
                types.Part.from_text(text="""請複述一遍照片中的訊息(只輸出照片中的訊息)"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        text += chunk.text  
    return text
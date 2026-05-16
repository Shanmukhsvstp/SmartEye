from groq import Groq
import base64
from paddleocr import PaddleOCR
import json

client = Groq(api_key="")


ocr = PaddleOCR(lang='en', enable_mkldnn=False)

file_name = "hw_t.jpg"

# Read image
with open(file_name, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")


result = ocr.predict(file_name)

texts = result[0]['rec_texts']
boxes = result[0]['rec_polys']

clean_ocr = []

for text, box in zip(texts, boxes):
    clean_ocr.append({
        "text": text,
        "x": int(box[0][0]),
        "y": int(box[0][1])
    })

ocr_text = json.dumps(clean_ocr)


completion = client.chat.completions.create(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    messages=[
        {
            "role": "system",
            "content": "You're a vision assistant who reads tables from images and returns CSV. Do not return any text or comments. The text might be human written, there might be empty cells, make sure to leave empty cells as empty strings and make sure the structure is perfect. For better accuracy, user will provide you the OCR data of the same input too."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the table as CSV."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }, 
                {
                    "type": "text",
                    "text": f"OCR DATA: \n{ocr_text}"
                }
            ]
        }
    ],
    temperature=0.2,
    max_completion_tokens=4096,
    stream=True,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
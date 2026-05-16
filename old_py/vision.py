from groq import Groq
import base64

client = Groq(api_key="")

# Read image
with open("ppr.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
# , there might be empty cells, make sure to leave empty cells as empty strings and make sure the structure is perfect. Usually the top left most cell is empty, for better accuracy, user will provide you the OCR data of the same input too.
completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "system",
            "content": "You're a vision assistant who reads tables from images and returns CSV. Do not return any text or comments. The text might be human written. Ignore the other data present in the input other than the table."
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
                }
            ]
        }
    ],
    temperature=0,
    max_completion_tokens=4096,
    stream=True
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
    
def detect(img):
    base64_image = base64.b64encode(img.read()).decode("utf-8")
# , there might be empty cells, make sure to leave empty cells as empty strings and make sure the structure is perfect. Usually the top left most cell is empty, for better accuracy, user will provide you the OCR data of the same input too.
    completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "system",
            "content": "You're a vision assistant who reads tables from images and returns CSV. Do not return any text or comments. The text might be human written. Ignore the other data present in the input other than the table."
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
                }
            ]
        }
    ],
    temperature=0,
    max_completion_tokens=4096,
    stream=True
)
    r=""
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")
        r+=chunk.choices[0].delta.content or ""
    return r
    
from groq import Groq
import base64
import cv2
import numpy as np
import json
from vision import detect
from google import genai
from google.genai import types

g_client = genai.Client(api_key="")

client = Groq(api_key="")
image = cv2.imread("ppr.jpg")

h, w = image.shape[:2]

# Remove top 30%
crop_start = int(h * 0.30)

image = image[crop_start:, :]

cv2.imwrite("temp.jpg", image)
# exit(0)
# Read image
with open("temp.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
# , there might be empty cells, make sure to leave empty cells as empty strings and make sure the structure is perfect. Usually the top left most cell is empty, for better accuracy, user will provide you the OCR data of the same input too.
completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "system",
            "content": 
                """
                You're a vision extraction model.

Your task:
1. Detect ONLY the marks table near the bottom-middle of the page.
2. This is specifically the SECOND TABLE FROM THE BOTTOM.
3. Ignore every other table in the image.
4. Return the 4 corner coordinates of ONLY this table.

Important spatial rules:
- The target table spans almost the full page width.
- Leave approximately 5% margin from left and right page edges.
- The table is located directly ABOVE the "Marks Obtained / Maximum Marks" section.
- The table usually contains columns like:
  - Question Number
  - Maximum Marks
  - Marks Obtained
  - TOTAL
- The height must tightly fit the table bounds.
- Include the full table border.

Coordinate rules:
- Coordinates must be image pixel coordinates.
- Return:
  - top_left
  - top_right
  - bottom_left
  - bottom_right

Also extract the student's USN:
- USN is exactly 10 characters.
- Example: 1BF25CS181

Return ONLY valid JSON.
Do not return markdown.
Do not explain anything.

Required JSON format:

{
  "USN": "1BF25CS181",
  "table_coordinates": {
    "top_left": [0, 0],
    "top_right": [0, 0],
    "bottom_left": [0, 0],
    "bottom_right": [0, 0]
  }
}
                """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the table"
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
response = ""
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
    response += chunk.choices[0].delta.content or ""

if response.startswith("```"):
    response_arr = response.splitlines()
    response_arr.pop(0); response_arr.pop(len(response_arr)-1)
    response = "\n".join(response_arr)
response = json.loads(response)
# print(f"RES:\n{response}")

top_left = response["table_coordinates"].get("top_left")
top_right = response["table_coordinates"].get("top_right")
bottom_left = response["table_coordinates"].get("bottom_left")
bottom_right = response["table_coordinates"].get("bottom_right")
top_left[1] += crop_start
top_right[1] += crop_start
bottom_left[1] += crop_start
bottom_right[1] += crop_start
image = cv2.imread("ppr.jpg")

image = cv2.resize(image, (768, 1365))

# Points in correct order
pts_src = np.array([
    top_left,
    top_right,
    bottom_right,
    bottom_left
], dtype=np.float32)

# Compute width
width_top = np.linalg.norm(np.array(top_right) - np.array(top_left))
width_bottom = np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))
width = int(max(width_top, width_bottom))

# Compute height
height_left = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
height_right = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
height = int(max(height_left, height_right))

# Destination rectangle
pts_dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)

# Perspective transform
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# Warp
image = cv2.warpPerspective(image, matrix, (width, height))

cv2.imwrite("cropped.jpg", image)



image = cv2.imread("cropped.jpg")

# Upscale
image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Denoise
gray = cv2.fastNlMeansDenoising(gray)

# Sharpen
blur = cv2.GaussianBlur(gray, (0, 0), 3)
sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

# Adaptive threshold
thresh = cv2.adaptiveThreshold(
    sharp,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    15
)

cv2.imwrite("temp.jpg", sharp)


with open("temp.jpg", "rb") as image_file:
    # final_res = detect(image_file)
    image_bytes = image_file.read()
    response = g_client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        ),
        "Identify and extract the table contents in this image."
    ]
)
    with open(".csv", "w") as csv:
        csv.write(str(response))
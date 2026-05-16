from google import genai
from google.genai import types
import cv2
import os

client = genai.Client(api_key="AIzaSyDXSJRLpXr9jFXPuGFskSbiIh1IIqJoBQM")

file = "ppr.jpg"
temp_file = "temp.jpg"

image = cv2.imread(file)

h, w = image.shape[:2]

# Remove top 40%
top_crop = int(h * 0.40)

# Remove bottom 25%
bottom_crop = int(h * 0.75)

# Keep middle section
image = image[top_crop:bottom_crop, :]

cv2.imwrite(temp_file, image)

# Read image
with open("temp.jpg", "rb") as image_file:
    image_bytes = image_file.read()
    # base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    response = client.models.generate_content(
    model="gemma-4-31b-it",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        ),
        """You're a vision assistant who reads tables from images and returns CSV. 
        Do not return any text or comments. The text might be human written. 
        Ignore the other data present in the input other than the table.
         
        """
    ],
    config=types.GenerateContentConfig(
        # Forces the model to allocate a high token budget for fine image details
        thinking_config=types.ThinkingConfig(thinking_level="high") 
    )
)
    print(response.text)
    with open(".csv", "w") as csv:
        csv.write(response.text) 
    

if os.path.exists(temp_file):
    os.remove(temp_file)
    print("File deleted successfully")
else:
    print("The file does not exist")
    
# , there might be empty cells, make sure to leave empty cells as empty strings and make sure the structure is perfect. Usually the top left most cell is empty, for better accuracy, user will provide you the OCR data of the same input too.

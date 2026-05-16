from google import genai
from google.genai import types
import cv2
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

__api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=__api_key)

file = "ppr.jpg"
temp_file = "temp.jpg"

prompt = """
You're a vision assistant who reads tables from images and returns CSV. 
Do not return any text or comments. The text might be human written. 
Ignore the other data present in the input other than the table.
"""


async def getCsv(nparr):
    # image = cv2.imread(input_image)
    image = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    return await getCsvFromCV(image=image)

    # h, w = image.shape[:2]

    # # Remove top 40%
    # top_crop = int(h * 0.40)

    # # Remove bottom 25%
    # bottom_crop = int(h * 0.75)

    # # Keep middle section
    # cropped_image = image[top_crop:bottom_crop, :]

    # # Convert image -> JPEG bytes
    # success, encoded_image = cv2.imencode(
    #     ".jpg",
    #     cropped_image
    # )

    # image_bytes = encoded_image.tobytes()

    # while True:

    #     try:

    #         response = client.models.generate_content(
    #             model="gemma-4-31b-it",
    #             contents=[
    #                 types.Part.from_bytes(
    #                     data=image_bytes,
    #                     mime_type="image/jpeg"
    #                 ),
    #                 prompt
    #             ]
    #         )

    #         return response.text

    #     except Exception as e:

    #         print(e)

    #         await asyncio.sleep(2)

    # return "Failed after retries"
    
    

async def getCsvFromCV(image):

    h, w = image.shape[:2]

    # Remove top 40%
    top_crop = int(h * 0.40)

    # Remove bottom 25%
    bottom_crop = int(h * 0.75)

    # Keep middle section
    cropped_image = image[top_crop:bottom_crop, :]

    # Convert image -> JPEG bytes
    success, encoded_image = cv2.imencode(
        ".jpg",
        cropped_image
    )

    image_bytes = encoded_image.tobytes()

    while True:

        try:

            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    ),
                    prompt
                ]
            )
            
            csv = response.text
            csv = csv.replace(
                "```csv",
                ""
            ).replace(
                "```",
                ""
            ).strip()

            return csv

        except Exception as e:

            print(e)

            await asyncio.sleep(2)

    # return "Failed after retries"
    
# , there might be empty cells, make sure to leave empty cells as empty strings and make sure the structure is perfect. Usually the top left most cell is empty, for better accuracy, user will provide you the OCR data of the same input too.

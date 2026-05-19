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

usn_prompt = """
    You're a vision assistant who can read "USN" from an image.
    You are given an unknowingly cropped image.
    There's a printed text labeled "USN" in the input image.
    Right beside the label, there are a few squared boxes where you can see human handwriting, a character each box.
    You should read the 10 character long string written in those boxes by human and return the text.
    Do not include your reasoning or thinking in the response.
    Make sure that it is accurate.
""".strip()


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
    
    top_start = int(h * 0.10)
    top_end = int(h * 0.50)

    top_section = image[top_start:top_end, :]

    # Convert image -> JPEG bytes
    success, encoded_image = cv2.imencode(
        ".jpg",
        cropped_image
    )
    success, usn_image = cv2.imencode(
        ".jpg",
        top_section
    )

    image_bytes = encoded_image.tobytes()
    usn_image_bytes = usn_image.tobytes()

    # table_data = run_gemma(prompt=prompt, image_bytes=image_bytes)
    
    table_data, usn_data = await asyncio.gather(
        run_gemma(prompt=prompt, image_bytes=image_bytes),
        run_gemini(usn_image_bytes=usn_image_bytes, usn_prompt=usn_prompt),
        return_exceptions=True
    )
    
    csv = table_data
    usn = usn_data.strip()
    
    csv = csv.replace(
            "```csv",
            ""
        ).replace(
            "```",
            ""
        ).strip()
    
    return csv, usn
    
            
async def run_gemma(prompt, image_bytes):

    while True:

        try:

            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemma-4-31b-it",
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    ),
                    prompt
                ]
            )
            
            if response and response.text:
                return response.text
            else:
                print("Received empty or malformed response text from Gemma. Retrying...")
                await asyncio.sleep(2)

        except Exception as e:

            print(f"gemma 31b: {e}")

            await asyncio.sleep(2)
    # return "Failed after retries"

async def run_gemini(usn_image_bytes, usn_prompt):

    while True:

        try:

            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemma-4-26b-a4b-it",
                contents=[
                    types.Part.from_bytes(
                        data=usn_image_bytes,
                        mime_type="image/jpeg"
                    ),
                    usn_prompt
                ]
            )
            if response and response.text:
                return response.text
            else:
                print("Received empty or malformed response text from Gemma. Retrying...")
                await asyncio.sleep(2)

        except Exception as e:

            print(f"gemma 26b: {e}")
            await asyncio.sleep(2)
# , there might be empty cells, make sure to leave empty cells as empty strings and make sure the structure is perfect. Usually the top left most cell is empty, for better accuracy, user will provide you the OCR data of the same input too.

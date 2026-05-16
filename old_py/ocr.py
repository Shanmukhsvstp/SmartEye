from paddleocr import PaddleOCR
import json


ocr = PaddleOCR(lang='en', enable_mkldnn=False)

result = ocr.predict("ppr.jpg")

texts = result[0]['rec_texts']
scores = result[0]['rec_scores']

with open("tab_res.json", "w") as file:
    json.dump(result, file, indent=4, default=str)

for text, score in zip(texts, scores):
    print(f"{text} ({score:.2f})")
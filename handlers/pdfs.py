import fitz
import numpy as np
import cv2

def extractImagesFromPDF(path):
    file = fitz.open(path)
    
    images = []
    
    for eachPage in range(len(file)):
        page = file[eachPage]
        img_bytes = page.get_pixmap().tobytes("png")
        npArr = np.frombuffer(
            img_bytes,
            np.uint8
        )
        
        image = cv2.imdecode(
            npArr,
            cv2.IMREAD_COLOR
        )
        
        images.append(image)
        
    return images
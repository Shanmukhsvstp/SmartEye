from handlers.ai import getCsv
import numpy as np


async def getCsvFromImage(img):
    nparr = np.frombuffer(img, np.uint8)
    data, usn = await getCsv(nparr=nparr)
    return data, usn
    

async def extractImagesFromPDF(pdf):
    images_list = list()
    return images_list
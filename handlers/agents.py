import uuid
import os
from handlers.pdfs import extractImagesFromPDF
from handlers.ai import getCsvFromCV

UPLOAD_DIR = "agent_uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


workers = {}


class Worker:
    def __init__(self, worker_id, path):
        self.worker_id = worker_id
        self.ws = None
        self.file_path = path
        
    def addWs(self, ws):
        self.ws = ws
    def getWs(self):
        return self.ws
    def cleanStorage(self):
        try:
            os.remove(self.file_path)
        except Exception as e:
            print(e)
    
    # def getUninitWorkerId(self):
    #     return self.worker_id
    
    # def addPdfToQueue(self, file_path):
    #     self.file_path = file_path
    
    # def addWorker(self):
    #     workers[self.worker_id] = self.ws
    #     return self.worker_id
    
    # def initWorker(self, ws):
    #     self.ws = ws
    #     self.addWorker()
    
    # def removeWorker(self):
    #     workers.pop(self.worker_id, None)
        
    # def getUser(self):
    #     return workers.get(self.worker_id, None)
    
    def getFilePath(self):
        return self.file_path
      
def createWorker():
    id = str(uuid.uuid4())
    pdf_path = f"{UPLOAD_DIR}/worker_{id}.pdf"
    worker = Worker(worker_id=id, path=pdf_path)
    workers[id] = worker
    return id, pdf_path

def initWorker(ws, worker_id):
    worker = workers[worker_id]
    worker.addWs(ws=ws)
    
def removeWorker(worker_id):
    worker = workers[worker_id]
    worker.cleanStorage()
    workers.pop(worker_id, None)

    
    
async def startAgent(worker_id):
    worker = workers.get(worker_id, None)
    if worker == None:
        return
    ws = worker.getWs()
    if (ws == None or worker.getFilePath() == None):
        return
    # Starting Agent
    await ws.send_json({
        "type": "progress",
        "progress": 0,
        "message": "Agent Started..."
    })
    file_path = worker.getFilePath()
    images = extractImagesFromPDF(path=file_path)
    CSVs = []
    tot = len(images)
    for i, image in enumerate(images):
        await ws.send_json({
            "type": "progress",
            "message": f"Analyzing page {i+1}",
            "progress": int((i/tot) * 100)
        })
        csv = await getCsvFromCV(image)
        CSVs.append(csv)
    removeWorker(worker_id=worker_id)
    cleanupStorage(worker_id=worker_id)
    await ws.send_json({
        "type": "task_complete",
        "message": "Completed Analysis",
        "data": CSVs
    })
from fastapi import FastAPI, Request, APIRouter, UploadFile, File, WebSocket
from fastapi.templating import Jinja2Templates
from handlers.files import getCsvFromImage
import uvicorn
from handlers.agents import startAgent, createWorker, UPLOAD_DIR, initWorker, removeWorker
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
port = os.getenv("PORT")
app = FastAPI()
api = APIRouter(prefix="/api")
scanner = APIRouter(prefix="/api/scan")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={
            "request": request
        }    
    )
    
    
@scanner.post("/image")
async def scan_image(img: UploadFile = File(...)):
    img_file = await img.read()
    print(img.content_type)
    print(len(img_file))
    data, usn = await getCsvFromImage(img_file)
    return {
        "message": "success",
        "data": data,
        "usn": usn
    }
    
@app.get("/scan_pdfs")
def agent(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="agent.html",
        context={
            "request": request
        }
    )
    
@scanner.post("/agent/upload")
async def uploadPdf(pdf: UploadFile = File(...)):
    pdf_file = await pdf.read()    
    worker_id, pdf_path = createWorker()
    with open(pdf_path, "wb") as file:
        file.write(pdf_file)
    return {
        "worker_id": worker_id
    }
    
@scanner.websocket("/agent/{worker_id}")
async def start_agent(ws: WebSocket, worker_id: str):
    await ws.accept()
    
    initWorker(ws=ws, worker_id=worker_id)
    await ws.send_json({
        "type": "connection_successful",
        "worker_id": f"{worker_id}"
    })
    asyncio.create_task(
        startAgent(worker_id=worker_id)
    )
    try:
        while True:
            data = await ws.receive_json()
    except:
        removeWorker(worker_id=worker_id)
        
app.include_router(api)
app.include_router(scanner)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
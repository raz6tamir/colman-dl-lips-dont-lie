import aiofiles
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from run_model import run_model

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    # Add any other allowed origins here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/info")
def site_home():
    return {
        "site": "Lips Don't Lie",
        "authors": "Raz Tamir, Naama Angel, Amit Hakmon, Adi Cohen Kashosh"
    }


@app.get("/words")
def get_words():
    return [
        'ACTION', 'CLOSE', 'HOSPITAL', 'LITTLE', 'NUMBER', 'PARTY', 'RESULT', 'SEVEN', 'TOMORROW', 'WALES'
    ]


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    # Perform video processing
    async with aiofiles.open("./video.mp4", 'wb') as out_file:
        content = await video.read()  # async read
        await out_file.write(content)  # async write

    prediction = run_model()

    # Send the processed video or message back to the frontend
    return prediction

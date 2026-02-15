import io
from fastapi import FastAPI
from starlette.responses import StreamingResponse

import mini_diffusion.sample as sample
app = FastAPI()


@app.get("/")
def root():
    
    return {"This is the api for the Diffusion Model"}



@app.get("/generate")
def generate():
    img_bytes = sample.sample()

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

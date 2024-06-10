import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

templates = Jinja2Templates(directory="templates")


class TextInput(BaseModel):
    text: str


@app.post("/analyze")
async def analyze_sentiment(input: TextInput):
        if input.text:
            result = sentiment_model(input.text)
            return result
        else:
            raise HTTPException(status_code=400,
                                detail="No text provided")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

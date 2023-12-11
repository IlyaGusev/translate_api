from typing import Union
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic_settings import BaseSettings
from pydantic import BaseModel

from translate import Translator


class Settings(BaseSettings):
    sp_model_path: str
    translator_model_path: str

settings = Settings()

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["translator"] = Translator(
        settings.sp_model_path,
        settings.translator_model_path
    )
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

class TranslationItem(BaseModel):
    text: str
    src_lang: str = "eng_Latn"
    tgt_lang: str = "rus_Cyrl"


@app.post("/translate")
def translate(item: TranslationItem) -> str:
    return ml_models["translator"].translate(item.text, item.src_lang, item.tgt_lang)

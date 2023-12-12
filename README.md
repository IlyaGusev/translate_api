1. Download models: `bash download.sh`
2. Install requirements: `pip install -r requirements.txt`
3. Install nltk punkt: `python3 -c "import nltk; nltk.download('punkt');"`
4. Run server: `SP_MODEL_PATH="models/flores200_sacrebleu_tokenizer_spm.model" TRANSLATOR_MODEL_PATH="models/nllb-200-3.3B-int8" uvicorn main:app --reload`

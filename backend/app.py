from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
import transformers
import torch
import os

app = FastAPI()

# Endpoint Data Models
class StringData(BaseModel):
    text: str
    length: int

@app.post("/summarize/")
async def summarize(data: StringData):
    """
    Submit an audio file

    This endpoint will take an audio file, generate a transcription, determine sentiment and return both
    """
    ########################################
    # Login to HuggingFace via an Access Token https://huggingface.co/settings/tokens
    ########################################
    login(token=os.getenv('HF_ACCESS_TOKEN'))

    # Check out model
    checkpoint = "Salesforce/codet5p-220m-bimodal"
    device = "cuda"  # for GPU usage or "cpu" for CPU usage

    # Use a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    # Use a pretrained model
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    # Get the text to summarize
    code = data.text

    # Create the tokenizer
    input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

    # Generate the output using the length provided
    generated_ids = model.generate(input_ids, max_length=text.length)
    print("Summary:")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
  
    return {"Summary": tokenizer.decode(generated_ids[0], skip_special_tokens=True)}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    details = f"{request.method} {request.url.path} - {response.status_code}"
    print(details)  # Use a proper logger in production
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

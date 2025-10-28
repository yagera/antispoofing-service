from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import os
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
import sys

sys.path.insert(0, '/app')
from shared import AntiSpoofingModel

UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_AGE_HOURS = 1

model = None


def cleanup_old_files():
    now = datetime.now()
    deleted_count = 0

    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age > timedelta(hours=MAX_FILE_AGE_HOURS):
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old files")


async def periodic_cleanup():
    while True:
        await asyncio.sleep(1800)
        cleanup_old_files()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading model...")
    model = AntiSpoofingModel("/app/weights/torch_script_weigths.pt")
    print("Model loaded successfully!")

    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield

    cleanup_task.cancel()
    print("Shutting down...")


app = FastAPI(
    title="Audio Anti-Spoofing API",
    description="API для обнаружения синтетической речи",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "message": "Audio Anti-Spoofing API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {allowed_extensions}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_filename = f"{timestamp}_{file.filename}"
    temp_path = UPLOAD_DIR / temp_filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model.predict(str(temp_path))

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "channels": len(results),
            "results": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                print(f"Error deleting temp file: {e}")


@app.post("/cleanup")
async def manual_cleanup():
    cleanup_old_files()
    return {"message": "Cleanup completed"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

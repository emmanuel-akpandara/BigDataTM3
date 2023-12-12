import os
from fastapi import FastAPI, File, UploadFile, HTTPException

from fastai.vision.all import *
from fastai.data.external import *
import random
import shutil

app = FastAPI()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the FastAI learner model from the exported file
learner = load_learner('Own_model/second_model.pkl')


def is_allowed_file(filename):
    """
    Check if the file extension is allowed.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for predicting the class of an image file.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        dict: A JSON response containing the predicted class.
    """
    if not file:
        raise HTTPException(
            status_code=400, detail="No image found in request.")

    if not is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400, detail=f"Invalid file type. Allowed types are {', '.join(ALLOWED_EXTENSIONS)}.")

    try:
        # Read the contents of the uploaded file
        contents = await file.read()
        # Create a FastAI image from the file contents
        img = PILImage.create(io.BytesIO(contents))
        # Make a prediction using the loaded learner model
        pred = learner.predict(img)
        print(pred)

        # If you just want the result:
        return {"prediction": str(pred[0])}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    # Run the FastAPI app using Uvicorn with auto-reload
    uvicorn.run("app:app", host='0.0.0.0', port=5050, reload=True)

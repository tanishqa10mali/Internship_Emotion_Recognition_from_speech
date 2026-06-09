from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import librosa
import keras
import time
import os

app = FastAPI(title="Speech Emotion Recognition API")

# Load the brain into memory when the server starts
model = keras.models.load_model("emotion_model.keras")
emotion_map = {0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    start_time = time.time() # Start the latency timer
    
    # 1. Save the incoming audio file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        # 2. Extract Acoustic Features (MFCCs)
        y, sr = librosa.load(temp_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        # 3. Reshape for the Conv1D model
        model_input = np.expand_dims(np.expand_dims(mfccs_processed, axis=0), axis=2)
        
        # 4. Run the Neural Network
        predictions = model.predict(model_input)[0]
        predicted_index = np.argmax(predictions)
        
        # 5. Format the strict minimal output for the dashboard
        latency = round((time.time() - start_time) * 1000)
        
        # Convert NumPy floats to standard Python floats for JSON conversion
        confidence_scores = {emotion_map[i]: round(float(predictions[i]) * 100, 2) for i in range(len(emotion_map))}
        
        return {
            "predicted_emotion": emotion_map[predicted_index].upper(),
            "confidence_scores": confidence_scores,
            "latency_ms": latency
        }
        
    finally:
        # Clean up the temporary audio file so we don't waste hard drive space
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py api.py gradio_app.py ./
COPY checkpoints/ checkpoints/

EXPOSE 8000

# Default: serve the FastAPI app.
# To run the Gradio app instead:
#   docker run -p 7860:7860 <image> python gradio_app.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Image Captioning App

An image captioning system that generates a short natural-language description from an uploaded image. The project contains the training notebook, a PyTorch inference implementation, a FastAPI service, and a Gradio web interface.

## Project Overview

The application uses a pretrained ResNet-50 to extract visual features and an LSTM-based decoder to generate a caption one token at a time. The trained captioning checkpoint is stored at `checkpoints/best_caption_model.pth` and is loaded once when the API or Gradio application starts.

The model automatically uses CUDA when it is available; otherwise it runs on the CPU.

## Dataset

Training uses the [Flickr8k dataset](https://github.com/jbrownlee/Datasets), which contains 8,091 images and five human-written captions per image. The notebook expects the Kaggle dataset layout:

```text
flickr8k/
	Images/
	captions.txt
```

Images are split by image, so captions belonging to the same image cannot cross data splits:

- 80% training images
- 10% validation images
- 10% test images

The split is shuffled with seed `42`. The dataset is not included in this repository; it is only needed to retrain or reevaluate the model.

## Architecture

```text
Input image
		-> RGB conversion and 224 x 224 normalization
		-> pretrained ResNet-50 without its classification layer
		-> 2048-dimensional feature vector
		-> Linear(2048, 256) + ReLU + dropout
		-> LSTM decoder
		-> vocabulary logits
		-> greedy next-token decoding
		-> generated caption
```

The decoder uses:

- 256-dimensional token embeddings
- One LSTM layer with hidden size 256
- Dropout of 0.4
- The encoded image as the initial hidden state
- A learned projection of the encoded image as the initial cell state
- A linear layer that maps each LSTM output to vocabulary scores

ResNet-50 is frozen during captioning. The learned encoder projection and LSTM decoder are loaded from the checkpoint.

## Preprocessing

### Images

Images are converted to RGB, resized to `224 x 224`, converted to tensors, and normalized with the ImageNet statistics used by the pretrained ResNet-50:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

During notebook preprocessing, ResNet features are extracted once and cached as `.pt` files. Training examples use the cached feature and caption sequence. Training-only feature noise with standard deviation `0.05` is used as lightweight augmentation; validation and test features are unchanged.

### Captions

Captions are:

1. Converted to lowercase.
2. Reduced to letters and spaces.
3. Normalized for repeated whitespace.
4. Wrapped with `<start>` and `<end>` tokens.
5. Tokenized by whitespace.
6. Converted to integer IDs.
7. Replaced with `<unk>` when a token is outside the vocabulary.
8. Padded to the maximum training-caption length using `<pad>`.

The vocabulary starts with `<pad>`, `<unk>`, `<start>`, and `<end>`. Words occurring fewer than two times in the training captions are excluded.

## Training Process

Training is implemented in `caption-generation-system.ipynb`.

- Batch size: `64`
- Maximum configured epochs: `20`
- Optimizer: Adam
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Loss: cross-entropy with `<pad>` ignored and label smoothing of `0.1`
- Gradient clipping: maximum norm `5.0`
- Scheduler: `ReduceLROnPlateau`, factor `0.5`, patience `3`
- Early stopping patience: `4` validation epochs without improvement

The decoder is trained with teacher forcing: the input is the caption sequence without its final token, and the target is the sequence without its initial token. After each epoch, the checkpoint with the lowest validation loss is saved as `best_caption_model.pth`.

## Evaluation Metrics and Results

The best checkpoint is evaluated once for every unique test image. Each generated caption is compared with all five reference captions. BLEU uses NLTK corpus scoring with method-4 smoothing. ROUGE-L uses the best F1 score among the five references for each image, and METEOR is averaged over the test set.

Results recorded in `evaluation_results.json`:

| Metric | Score |
| --- | ---: |
| BLEU-1 | 0.5734 |
| BLEU-2 | 0.3971 |
| BLEU-3 | 0.2621 |
| BLEU-4 | 0.1696 |
| ROUGE-L | 0.4427 |
| METEOR | 0.3898 |

These scores are reference-based language metrics. They should be interpreted together with qualitative inspection of captions because multiple descriptions can be valid for the same image.

## Installation

Python 3.11 is recommended. From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first model startup may download the pretrained ResNet-50 weights through `torchvision` if they are not already cached locally.

## Run the Application

### FastAPI

Start the API on port `8000`:

```powershell
uvicorn api:app --reload
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

### Gradio

Start the browser interface on port `7860`:

```powershell
python gradio_app.py
```

Open `http://127.0.0.1:7860` and upload an image. The interface accepts a PIL-compatible image and displays the generated caption.

### Docker

Build and run the default FastAPI service:

```powershell
docker build -t image-captioning-app .
docker run --rm -p 8000:8000 image-captioning-app
```

To run the Gradio interface from the same image:

```powershell
docker run --rm -p 7860:7860 image-captioning-app python gradio_app.py
```

## API Usage

### Health check

```powershell
curl http://127.0.0.1:8000/health
```

Response:

```json
{"status":"ok"}
```

### Generate a caption

Send an image as multipart form data to `POST /predict`:

```powershell
curl -X POST http://127.0.0.1:8000/predict -F "file=@examples/park.jpg"
```

Example response:

```json
{"caption":"a dog is running through the grass"}
```

The exact caption depends on the input image. The endpoint returns HTTP 400 when the uploaded file is not identified as an image or cannot be opened by Pillow.

## Example Input and Output

Example input: a JPEG or PNG photograph of a dog running outdoors, uploaded through the Gradio image control or as `examples/park.jpg` in the API request.

Example generated output:

```text
a dog is running through the grass
```

Generated text is produced with greedy decoding. Generation starts with `<start>` and stops when `<end>` is predicted or when the checkpoint's maximum caption length is reached.

## Project Files

| File | Purpose |
| --- | --- |
| `caption-generation-system.ipynb` | Dataset preparation, feature extraction, training, and evaluation |
| `model.py` | Checkpoint loading, image preprocessing, and inference |
| `api.py` | FastAPI health and prediction endpoints |
| `gradio_app.py` | Gradio upload interface |
| `checkpoints/best_caption_model.pth` | Trained model weights and vocabulary metadata |
| `evaluation_results.json` | Recorded test-set metrics |
| `Dockerfile` | Container image definition |
| `requirements.txt` | Python dependencies |

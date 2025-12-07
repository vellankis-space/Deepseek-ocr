# DeepSeek OCR Demo

A simple demo application that lets you run **DeepSeek-OCR** on PDFs and images to extract text quickly and easily.

The app is built with `Gradio` and exposes a web UI where you can upload one or more documents and view the recognized text.

## Features

- **PDF support** – Upload PDF documents and extract text from pages.
- **Image support** – Upload images (e.g. PNG, JPG) and run OCR on them.
- **Web UI with Gradio** – No need to write code to test the model; just use the browser.
- **Configurable model/backend** – Logic lives in `app.py` so you can easily swap models or tweak behavior.

## Project Structure

- `app.py` – Main Gradio app entry point.
- `examples/` – Example PDFs/images you can use for quick testing.
- `requirements.txt` – Python dependencies for running the demo.
- `.gitignore` – Git ignore rules (e.g. `venv/`, caches, examples, etc.).

## Setup

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate  # Windows (PowerShell or CMD)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the App

From the project root:

```bash
python app.py
```

Gradio will print a local URL in the terminal (something like `http://127.0.0.1:7860`). Open it in your browser to use the OCR demo.

## Usage

1. Open the Gradio URL in your browser.
2. Upload a PDF or image file.
3. Wait for the model to run OCR on your document.
4. Copy or download the extracted text as needed.

## Notes

- Make sure you have a compatible Python version installed (see `requirements.txt`).
- If you run this on a remote machine or within a container, you may need to expose the Gradio port publicly.

## License

This project is provided under the MIT license (see repository settings or add a `LICENSE` file if needed).
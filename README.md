# Vintern-1B Image-to-Text Demo

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](docs/CHANGELOG.md)
[![Docker](https://img.shields.io/badge/docker-namnhcntt%2Fvintern--image--to--text-blue.svg)](https://hub.docker.com/r/namnhcntt/vintern-image-to-text)

FastAPI and Docker demo for [5CD-AI/Vintern-1B-v3_5](https://huggingface.co/5CD-AI/Vintern-1B-v3_5), a Vietnamese-focused vision-language model for OCR, document understanding, and image-to-text extraction.

Vietnamese documentation is available in [README.vi.md](README.vi.md).

## Features

- Web UI for image/PDF upload and OCR-style extraction
- Prompt customization and structured output templates
- JSON API and multipart upload API
- Optional streaming responses
- Automatic hardware detection for CPU, Apple Silicon MPS, and NVIDIA CUDA
- Docker build variants for AMD64, ARM64, and CUDA

## Project Layout

```text
.
├── app.py                  # FastAPI application
├── frontend/               # Web UI
├── docker/                 # Dockerfile variants
├── scripts/                # Setup, run, and Docker helper scripts
├── tests/                  # Manual test scripts and fixtures
├── docs/                   # Changelog and supplemental docs
└── tools/mcp/              # Optional Playwright MCP package files
```

## Requirements

- Python 3.8+
- pip and venv
- Docker, if you want to use containers
- Enough disk space for Python dependencies and Hugging Face model cache

## Run Locally

Install dependencies and start the API:

```bash
pip install -r requirements.txt
python app.py
```

The app will be available at:

- Web UI: http://localhost:8000
- API info: http://localhost:8000/api

You can also use the helper script, which creates and uses a local virtual environment:

```bash
chmod +x scripts/run_demo.sh
./scripts/run_demo.sh
```

For explicit virtual environment setup:

```bash
chmod +x scripts/setup_venv.sh
./scripts/setup_venv.sh
```

On Apple Silicon, this helper installs compatible dependencies and enables MPS fallback where needed:

```bash
chmod +x scripts/fix_transformers.sh
./scripts/fix_transformers.sh
```

## Docker

Build the best image for the current machine:

```bash
chmod +x scripts/build_docker.sh
./scripts/build_docker.sh
```

Run the container:

```bash
chmod +x scripts/run_docker.sh
./scripts/run_docker.sh
```

Manual build using the default Dockerfile:

```bash
docker build -f docker/Dockerfile -t namnhcntt/vintern-image-to-text:latest .
```

Manual run:

```bash
docker run --name vintern-demo -p 8000:8000 -d namnhcntt/vintern-image-to-text:latest
```

To reuse the Hugging Face cache between container runs:

```bash
docker run \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8000:8000 \
  namnhcntt/vintern-image-to-text:latest
```

Build specific architecture images:

```bash
./scripts/build_multiarch_docker.sh --arm64
./scripts/build_multiarch_docker.sh --amd64
./scripts/build_multiarch_docker.sh --cuda
./scripts/build_multiarch_docker.sh --all
```

Add `--push` to publish the generated images and manifests to Docker Hub.

## Configuration

The app reads these environment variables:

```bash
export VINTERN_MODEL_ID=5CD-AI/Vintern-1B-v3_5
export VINTERN_STREAM=false
export VINTERN_MAX_TOKENS=1024
export VINTERN_TEMPERATURE=0.0
export VINTERN_DO_SAMPLE=false
export VINTERN_NUM_BEAMS=3
export VINTERN_REPETITION_PENALTY=2.5
```

Docker example:

```bash
VINTERN_STREAM=true ./scripts/run_docker.sh
```

## API

`GET /api`

Returns basic service information.

`GET /api/templates`

Lists available structured prompt templates.

`POST /api/image-to-text`

Accepts JSON with a base64 image payload:

```json
{
  "prompt": "Extract the key information from this image and return Markdown.",
  "image": "base64_encoded_image_data",
  "max_tokens": 1024,
  "temperature": 0.0,
  "do_sample": false,
  "num_beams": 3,
  "repetition_penalty": 2.5,
  "stream": false,
  "template": "markdown"
}
```

`POST /api/upload-image`

Accepts multipart form data with an uploaded image or PDF plus prompt options.

## Tests And Fixtures

The repo includes manual API smoke tests under `tests/`. Start the server first, then run:

```bash
python tests/test_quick.py
python tests/test_pdf_upload.py
```

To regenerate the PDF fixture:

```bash
python tests/create_test_pdf.py
```

## Troubleshooting

If port `8000` is already in use, run `python app.py --port 8080` or map Docker to another host port:

```bash
docker run --name vintern-demo -p 8080:8000 -d namnhcntt/vintern-image-to-text:latest
```

If the container does not start, check logs:

```bash
docker logs vintern-demo
```

For Apple Silicon MPS errors, set:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## License

See [LICENSE](LICENSE). The Vintern model is distributed under the terms published by 5CD-AI on the model's Hugging Face page.

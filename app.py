import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import AsyncGenerator, Iterator
import pathlib
import base64
import io
from pydantic import BaseModel
from typing import Optional, List
import argparse
import sys
import os
import socket
import asyncio
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import tempfile

app = FastAPI()

# Add CORS middleware to allow requests from web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend URL instead of *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Environment variables and configuration
def get_env_bool(name, default=False):
    """Get a boolean value from environment variable"""
    value = os.environ.get(name, str(default)).lower()
    return value in ('true', '1', 'yes', 'y', 'on')

def get_env_int(name, default=0):
    """Get an integer value from environment variable"""
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default

def get_env_float(name, default=0.0):
    """Get a float value from environment variable"""
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default

def get_env_str(name, default=""):
    """Get a string value from environment variable"""
    return os.environ.get(name, default)

# Default configuration
DEFAULT_STREAM = get_env_bool("VINTERN_STREAM", False)
DEFAULT_MAX_TOKENS = get_env_int("VINTERN_MAX_TOKENS", 1024)
DEFAULT_TEMPERATURE = get_env_float("VINTERN_TEMPERATURE", 0.0)
DEFAULT_DO_SAMPLE = get_env_bool("VINTERN_DO_SAMPLE", False)
DEFAULT_NUM_BEAMS = get_env_int("VINTERN_NUM_BEAMS", 3)
DEFAULT_REPETITION_PENALTY = get_env_float("VINTERN_REPETITION_PENALTY", 2.5)

# Constants from the Vintern-1B-v3_5 model
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_ID = get_env_str("VINTERN_MODEL_ID", "5CD-AI/Vintern-1B-v3_5")

# Model and tokenizer instances
model = None
tokenizer = None

def get_best_device_and_dtype():
    """
    Determine the best available device and appropriate dtype based on hardware priorities:
    - For Mac: Apple Silicon MPS > NVIDIA/AMD GPU > CPU
    - For Windows/Linux: NVIDIA/AMD GPU > CPU
    """
    device = None
    dtype = None
    device_name = None

    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        device_name = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"

    # Check for MPS (Apple Silicon)
    # On Apple Silicon, prefer MPS over CUDA if both are available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check if we're on macOS and arm64 architecture
        import platform
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            device = torch.device("mps")
            # Use float16 for MPS as bfloat16 might not be fully supported
            dtype = torch.float16
            device_name = "Apple Silicon MPS"

    # If no GPU is available, use CPU
    if device is None:
        device = torch.device("cpu")
        # Use float32 for CPU for better accuracy
        dtype = torch.float32
        device_name = "CPU"

    return device, dtype, device_name

def load_model():
    global model, tokenizer

    if model is None or tokenizer is None:
        print("Loading model and tokenizer...")

        # Get the best device and dtype
        device, dtype, device_name = get_best_device_and_dtype()
        print(f"Using device: {device_name}")

        # Check if MPS fallback is enabled
        if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            print("MPS fallback is enabled for unsupported operations")

        # Pin model version to avoid downloading new files
        model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            # low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=(device.type == "cuda"),  # Only use flash attention on CUDA
            revision="main",  # Pin to main branch
        ).eval()

        # Move to appropriate device
        model = model.to(device)
        print(f"Model loaded on {device_name}")

        # Pin tokenizer version to the same revision
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            use_fast=False,
            revision="main"  # Pin to main branch
        )
        print("Model and tokenizer loaded successfully")

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD

    # Check if running on MPS (Apple Silicon) and PYTORCH_ENABLE_MPS_FALLBACK is not set
    is_mps = (hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and
              os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1")

    # Use BILINEAR interpolation instead of BICUBIC for MPS compatibility if fallback not enabled
    interpolation_mode = InterpolationMode.BILINEAR if is_mps else InterpolationMode.BICUBIC

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=interpolation_mode),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def convert_pdf_to_images(pdf_data):
    """Convert PDF bytes to list of PIL Images"""
    try:
        # Try using pdf2image first (better quality)
        images = convert_from_bytes(pdf_data, dpi=200, fmt='RGB')
        return images
    except Exception as e:
        print(f"pdf2image failed, trying PyMuPDF: {e}")
        try:
            # Fallback to PyMuPDF
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render page to image with higher resolution
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data)).convert('RGB')
                images.append(image)
            doc.close()
            return images
        except Exception as e2:
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e2)}")

def process_image(image_data, input_size=448, max_num=6):
    """Process image from bytes or base64 string"""
    # Convert base64 to image if needed
    if isinstance(image_data, str) and image_data.startswith('data:image'):
        # Extract the base64 content after the comma
        image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)

    # Open image from bytes
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Apply transformations
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)

    # Get the best device and dtype
    device, dtype, _ = get_best_device_and_dtype()

    # Convert to appropriate dtype and move to device
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    return pixel_values

def process_pdf(pdf_data, input_size=448, max_num=6):
    """Process PDF by converting to images and then processing each page"""
    images = convert_pdf_to_images(pdf_data)
    all_pixel_values = []

    for image in images:
        # Convert PIL image to bytes for processing
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Process each page as an image
        pixel_values = process_image(img_byte_arr, input_size, max_num)
        all_pixel_values.append(pixel_values)

    return all_pixel_values, len(images)

# Prompt templates for structured data extraction
PROMPT_TEMPLATES = {
    "json": """Trích xuất thông tin từ hình ảnh và trả về dưới dạng JSON có cấu trúc.
Hãy phân tích nội dung và tổ chức thông tin một cách logic.
Trả về JSON hợp lệ với các trường phù hợp:

```json
{
  "title": "Tiêu đề hoặc chủ đề chính",
  "content": "Nội dung chính",
  "details": {
    "key1": "value1",
    "key2": "value2"
  },
  "metadata": {
    "type": "loại tài liệu",
    "language": "ngôn ngữ",
    "confidence": "độ tin cậy"
  }
}
```""",

    "yaml": """Trích xuất thông tin từ hình ảnh và trả về dưới dạng YAML có cấu trúc.
Hãy phân tích nội dung và tổ chức thông tin một cách logic.
Trả về YAML hợp lệ:

```yaml
title: "Tiêu đề hoặc chủ đề chính"
content: "Nội dung chính"
details:
  key1: "value1"
  key2: "value2"
metadata:
  type: "loại tài liệu"
  language: "ngôn ngữ"
  confidence: "độ tin cậy"
```""",

    "markdown": """Trích xuất thông tin từ hình ảnh và trả về dưới dạng Markdown có cấu trúc.
Hãy phân tích nội dung và tổ chức thông tin một cách logic với các heading, list, table phù hợp.

Sử dụng cấu trúc Markdown như:
- # Tiêu đề chính
- ## Tiêu đề phụ
- **Văn bản in đậm**
- *Văn bản in nghiêng*
- - Danh sách
- | Bảng | Dữ liệu |
- > Trích dẫn
- `Code` hoặc ```code block```""",

    "table": """Trích xuất dữ liệu dạng bảng từ hình ảnh và trả về dưới dạng Markdown table.
Nếu có nhiều bảng, hãy tách riêng từng bảng.
Sử dụng format:

| Cột 1 | Cột 2 | Cột 3 |
|-------|-------|-------|
| Dữ liệu 1 | Dữ liệu 2 | Dữ liệu 3 |""",

    "invoice": """Trích xuất thông tin hóa đơn từ hình ảnh và trả về dưới dạng JSON có cấu trúc:

```json
{
  "invoice_info": {
    "number": "số hóa đơn",
    "date": "ngày tháng",
    "due_date": "hạn thanh toán"
  },
  "seller": {
    "name": "tên người bán",
    "address": "địa chỉ",
    "tax_id": "mã số thuế",
    "phone": "số điện thoại"
  },
  "buyer": {
    "name": "tên người mua",
    "address": "địa chỉ",
    "tax_id": "mã số thuế"
  },
  "items": [
    {
      "description": "mô tả sản phẩm",
      "quantity": "số lượng",
      "unit_price": "đơn giá",
      "total": "thành tiền"
    }
  ],
  "totals": {
    "subtotal": "tổng tiền hàng",
    "tax": "thuế VAT",
    "total": "tổng cộng"
  }
}
```""",

    "form": """Trích xuất thông tin từ form/biểu mẫu trong hình ảnh và trả về dưới dạng JSON:

```json
{
  "form_type": "loại biểu mẫu",
  "fields": {
    "field_name_1": "giá trị 1",
    "field_name_2": "giá trị 2",
    "checkbox_field": true/false,
    "date_field": "YYYY-MM-DD"
  },
  "signatures": ["vị trí chữ ký nếu có"],
  "stamps": ["vị trí con dấu nếu có"]
}
```"""
}

class ImageToTextRequest(BaseModel):
    prompt: str
    image: Optional[str] = None  # Base64 encoded image
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    do_sample: Optional[bool] = DEFAULT_DO_SAMPLE
    num_beams: Optional[int] = DEFAULT_NUM_BEAMS
    repetition_penalty: Optional[float] = DEFAULT_REPETITION_PENALTY
    stream: Optional[bool] = DEFAULT_STREAM  # Whether to stream the response
    template: Optional[str] = None  # Template type for structured output

@app.on_event("startup")
async def startup_event():
    # Load model at startup
    try:
        load_model()
        print("Model loaded successfully at startup")
    except Exception as e:
        print(f"Warning: Failed to load model at startup: {e}")
        print("Model will be loaded on first request")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the index.html file"""
    try:
        # First try to serve from the root directory
        root_index_path = pathlib.Path(__file__).parent / "index.html"
        if root_index_path.exists():
            with open(root_index_path, "r", encoding="utf-8") as f:
                return f.read()

        # Fallback to static directory for Docker compatibility
        index_path = static_dir / "index.html"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"Error reading index.html: {e}")
        # Return a simple HTML page instead of a dict to avoid encoding error

    # Return a simple HTML page instead of a dict to avoid encoding error
    return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vintern-1B Image-to-Text API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #333; }
                .container { max-width: 800px; margin: 0 auto; }
                .info { background-color: #f5f5f5; padding: 20px; border-radius: 5px; }
                a { color: #0066cc; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Vintern-1B Image-to-Text API</h1>
                <div class="info">
                    <p>The API is running successfully. You can access the following endpoints:</p>
                    <ul>
                        <li><strong>API Information:</strong> <a href="/api">/api</a></li>
                        <li><strong>Image to Text:</strong> <code>/api/image-to-text</code> (POST)</li>
                        <li><strong>Upload Image:</strong> <code>/api/upload-image</code> (POST)</li>
                    </ul>
                    <p>For more information, please refer to the documentation.</p>
                </div>
            </div>
        </body>
        </html>
        """

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {"message": "Vintern-1B Image-to-Text API is running", "version": "1.0.0"}

@app.get("/api/templates")
async def get_templates():
    """Get available prompt templates"""
    return {
        "templates": {
            name: {
                "name": name,
                "description": f"Template for {name} structured output"
            }
            for name in PROMPT_TEMPLATES.keys()
        }
    }

async def generate_streaming_response(
    pixel_values,
    prompt,
    generation_config,
    model,
    tokenizer
) -> AsyncGenerator[str, None]:
    """Generate streaming response from the model"""
    try:
        # This is a placeholder for actual streaming implementation
        # The actual implementation depends on the model's streaming capabilities

        # For now, we'll simulate streaming by yielding chunks of the response
        # In a real implementation, you would use the model's streaming API

        # Generate the full response first
        response, _ = model.chat(
            tokenizer,
            pixel_values,
            prompt,
            generation_config,
            history=None,
            return_history=True
        )

        # Simulate streaming by yielding chunks
        # In a real implementation, you would yield tokens as they are generated
        chunk_size = 10  # Characters per chunk
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i+chunk_size]
            yield f'data: {{"text": "{chunk}"}}\n\n'
            await asyncio.sleep(0.05)  # Simulate generation time

        # Send a final empty data message to signal the end of the stream
        yield 'data: [DONE]\n\n'

    except Exception as e:
        yield f'data: {{"error": "{str(e)}"}}\n\n'
        yield 'data: [DONE]\n\n'

@app.post("/api/image-to-text")
async def image_to_text(request: ImageToTextRequest):
    try:
        # Ensure model is loaded
        if model is None or tokenizer is None:
            load_model()

        # Process the image
        if not request.image:
            raise HTTPException(status_code=400, detail="Image data is required")

        pixel_values = process_image(request.image)

        # Prepare prompt with template if specified
        final_prompt = request.prompt
        if request.template and request.template in PROMPT_TEMPLATES:
            template_prompt = PROMPT_TEMPLATES[request.template]
            final_prompt = f"{template_prompt}\n\nYêu cầu bổ sung: {request.prompt}" if request.prompt.strip() else template_prompt

        # Configure generation parameters
        generation_config = {
            "max_new_tokens": request.max_tokens,
            "do_sample": request.do_sample,
            "num_beams": request.num_beams,
            "repetition_penalty": request.repetition_penalty
        }

        # Check if streaming is requested
        if request.stream:
            # Return a streaming response
            return StreamingResponse(
                generate_streaming_response(
                    pixel_values,
                    final_prompt,
                    generation_config,
                    model,
                    tokenizer
                ),
                media_type="text/event-stream"
            )
        else:
            # Generate text from image (non-streaming)
            response, _ = model.chat(
                tokenizer,
                pixel_values,
                final_prompt,
                generation_config,
                history=None,
                return_history=True
            )

            return {"text": response, "template_used": request.template}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# For file uploads (alternative endpoint)
@app.post("/api/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    prompt: str = Form("Trích xuất thông tin chính trong ảnh và trả về dạng markdown."),
    max_tokens: int = Form(DEFAULT_MAX_TOKENS),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    do_sample: bool = Form(DEFAULT_DO_SAMPLE),
    num_beams: int = Form(DEFAULT_NUM_BEAMS),
    repetition_penalty: float = Form(DEFAULT_REPETITION_PENALTY),
    stream: bool = Form(DEFAULT_STREAM),
    template: str = Form("")
):
    try:
        # Ensure model is loaded
        if model is None or tokenizer is None:
            load_model()

        # Read the uploaded file
        file_data = await file.read()

        # Check file type and process accordingly
        if file.content_type == "application/pdf" or file.filename.lower().endswith('.pdf'):
            # Process PDF file
            all_pixel_values, num_pages = process_pdf(file_data)

            # Prepare prompt with template if specified
            final_prompt = prompt
            print(f"DEBUG: template = '{template}', type = {type(template)}")
            print(f"DEBUG: template in PROMPT_TEMPLATES = {template in PROMPT_TEMPLATES if template else False}")
            if template and template.strip() and template in PROMPT_TEMPLATES:
                template_prompt = PROMPT_TEMPLATES[template]
                final_prompt = f"{template_prompt}\n\nYêu cầu bổ sung: {prompt}" if prompt.strip() else template_prompt
                print(f"DEBUG: Using template '{template}'")
            else:
                print(f"DEBUG: No template used")

            # Process each page and combine results
            all_responses = []
            for i, pixel_values in enumerate(all_pixel_values):
                page_prompt = f"Trang {i+1}/{num_pages}: {final_prompt}"

                # Configure generation parameters
                generation_config = {
                    "max_new_tokens": max_tokens,
                    "do_sample": do_sample,
                    "num_beams": num_beams,
                    "repetition_penalty": repetition_penalty
                }

                # Generate text from image (non-streaming for PDF)
                response, _ = model.chat(
                    tokenizer,
                    pixel_values,
                    page_prompt,
                    generation_config,
                    history=None,
                    return_history=True
                )
                all_responses.append(f"## Trang {i+1}\n\n{response}")

            # Combine all pages
            combined_response = "\n\n".join(all_responses)
            template_used = template if template and template.strip() and template in PROMPT_TEMPLATES else None
            return {"text": combined_response, "pages": num_pages, "template_used": template_used}

        else:
            # Process as image
            pixel_values = process_image(file_data)

            # Prepare prompt with template if specified
            final_prompt = prompt
            if template and template.strip() and template in PROMPT_TEMPLATES:
                template_prompt = PROMPT_TEMPLATES[template]
                final_prompt = f"{template_prompt}\n\nYêu cầu bổ sung: {prompt}" if prompt.strip() else template_prompt

            # Configure generation parameters
            generation_config = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty
            }

            # Check if streaming is requested
            if stream:
                # Return a streaming response
                return StreamingResponse(
                    generate_streaming_response(
                        pixel_values,
                        final_prompt,
                        generation_config,
                        model,
                        tokenizer
                    ),
                    media_type="text/event-stream"
                )
            else:
                # Generate text from image (non-streaming)
                response, _ = model.chat(
                    tokenizer,
                    pixel_values,
                    final_prompt,
                    generation_config,
                    history=None,
                    return_history=True
                )

                template_used = template if template and template.strip() and template in PROMPT_TEMPLATES else None
                return {"text": response, "template_used": template_used}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def find_available_port(start_port=8000, max_port=9000):
    """Find an available port by checking if it's already in use"""
    port = start_port
    while port <= max_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"No available ports found between {start_port} and {max_port}")

def run_server(host='0.0.0.0', port=8000, reload=False):
    """Run the FastAPI server"""
    import uvicorn

    # Suppress the urllib3 warning
    import warnings
    warnings.filterwarnings("ignore", category=Warning)

    try:
        uvicorn.run("app:app", host=host, port=port, reload=reload)
    except OSError as e:
        print(f"Error: {e}")
        print(f"Port {port} already in use. Attempting to find an available port...")
        try:
            available_port = find_available_port(port + 1)
            print(f"Found available port: {available_port}")
            uvicorn.run("app:app", host=host, port=available_port, reload=reload)
        except Exception as e:
            print(f"Failed to start server: {e}")
            sys.exit(1)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start the Vintern-1B Image-to-Text API server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload on code changes')

    args = parser.parse_args()

    # Run the server
    run_server(host=args.host, port=args.port, reload=args.reload)
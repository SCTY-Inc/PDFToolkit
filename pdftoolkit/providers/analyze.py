"""Image/chart analysis providers."""

from pathlib import Path

import torch
from PIL import Image

from pdftoolkit.clients import get_ollama_client, get_together_client, api_retry
from pdftoolkit.utils import get_device, image_to_base64, image_to_base64_raw

# Cached ColQwen model (lazy-loaded)
_colqwen_model = None
_colqwen_processor = None


def _get_colqwen():
    """Get cached ColQwen model and processor (lazy-loaded)."""
    global _colqwen_model, _colqwen_processor
    if _colqwen_model is None:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        device = get_device()
        _colqwen_model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v0.1", torch_dtype=torch.bfloat16, device_map=device
        ).eval()
        _colqwen_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
    return _colqwen_model, _colqwen_processor


def _score_image(image: Image.Image, queries: list[str]) -> list[float]:
    """Score image against queries using ColQwen."""
    model, processor = _get_colqwen()
    batch_images = processor.process_images([image]).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    return [float(s[0]) for s in scores]


@api_retry
def analyze_ollama(input_path: Path, query: str) -> str:
    """Analyze an image with Llama Vision via local Ollama."""
    client = get_ollama_client()
    image_b64 = image_to_base64_raw(input_path)

    response = client.chat(
        model="llama3.2-vision:11b",
        messages=[{"role": "user", "content": query, "images": [image_b64]}],
    )

    return response["message"]["content"]


@api_retry
def analyze_together(input_path: Path, query: str, threshold: float = 0.5) -> str:
    """Analyze image with Together API, using ColQwen for confidence filtering."""
    image = Image.open(input_path)
    confidence = _score_image(image, ["This is a quantitative data visualization"])[0]

    if confidence < threshold:
        return f"Low confidence ({confidence:.3f}) that this is a data visualization. Skipping analysis."

    img_base64 = image_to_base64(input_path)
    client = get_together_client()

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You are a data analyst. {query}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            }
        ],
        max_tokens=300,
        temperature=0.2,
    )

    return f"[Confidence: {confidence:.3f}]\n{response.choices[0].message.content}"


def analyze_colqwen(input_path: Path, queries: list[str], threshold: float = 0.5) -> list[tuple[str, float]]:
    """Find relevant content in image using ColQwen embedding scores."""
    image = Image.open(input_path)
    scores = _score_image(image, queries)

    results = [(q, s) for q, s in zip(queries, scores)]
    results.sort(key=lambda x: x[1], reverse=True)

    return [(q, s) for q, s in results if s >= threshold]

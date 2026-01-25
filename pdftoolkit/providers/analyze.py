"""Image/chart analysis providers."""

from pathlib import Path

import torch
from PIL import Image

from pdftoolkit.clients import get_ollama_client, get_together_client, api_retry
from pdftoolkit.utils import get_device, image_to_base64, image_to_base64_raw


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
    """Analyze image with Together API, optionally using ColQwen for confidence filtering."""
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    device = get_device()

    # Load ColQwen for confidence scoring
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1", torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

    # Get confidence score
    image = Image.open(input_path)
    batch_images = processor.process_images([image]).to(model.device)
    batch_queries = processor.process_queries(
        ["This is a quantitative data visualization"]
    ).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    confidence = scores[0][0]

    if confidence < threshold:
        return f"Low confidence ({confidence:.3f}) that this is a data visualization. Skipping analysis."

    # High confidence - analyze with Together API
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
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    device = get_device()

    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1", torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

    image = Image.open(input_path)
    batch_images = processor.process_images([image]).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)

    results = [(query, float(score[0])) for query, score in zip(queries, scores)]
    results.sort(key=lambda x: x[1], reverse=True)

    # Filter by threshold
    return [(q, s) for q, s in results if s >= threshold]

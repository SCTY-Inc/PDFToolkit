import torch
from pathlib import Path
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor

from pdftoolkit.clients import get_together_client, api_retry
from pdftoolkit.utils import get_device, image_to_base64

# Device setup
device = get_device()
print(f"Using device: {device}")

# Load models
model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1", torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")


@api_retry
def analyze_image(image_path, query):
    """Analyze image with ColQwen2 confidence check, then Llama if confident."""
    # Get confidence score for quantitative analysis
    image = Image.open(image_path)
    batch_images = processor.process_images([image]).to(model.device)
    batch_queries = processor.process_queries(
        ["This is a quantitative data visualization"]
    ).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    confidence = scores[0][0]
    print(f"Confidence this is a data viz: {confidence:.4f}")

    # If high confidence, analyze details
    if confidence > 0.5:
        img_base64 = image_to_base64(image_path)
        client = get_together_client()
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are a data analyst. " + query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content
    else:
        return "Warning: This image may not be a data visualization. Please verify the content."


# Image to analyze
image_path = Path("output/ACCENTURE-Marker/_page_7_Figure_6.jpeg")

queries = [
    "What type of visualization is this and what are its key metrics?",
    "What are the main quantitative insights and trends shown in this data?",
    "Summarize the key message of this visualization in 2-3 concise bullet points.",
]

print("=== Quantitative Analysis with Confidence Scoring ===")
for query in queries:
    print(f"\nQ: {query}")
    print(f"A: {analyze_image(image_path, query)}")

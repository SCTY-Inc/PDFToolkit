import torch
from pathlib import Path
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor

from pdftoolkit.utils import get_device

# Device setup
device = get_device()
print(f"Using device: {device}")

# Load models
model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1", torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")


def find_relevant_content(image_path, search_queries):
    """Find relevant content in the image based on search queries."""
    # Load image
    image = Image.open(image_path)

    # Process inputs
    batch_images = processor.process_images([image]).to(model.device)
    batch_queries = processor.process_queries(search_queries).to(model.device)

    # Get embeddings
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    # Get relevance scores
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)

    # Sort and return results
    results = list(zip(search_queries, scores))
    results.sort(key=lambda x: x[1][0], reverse=True)
    return results


# Image to analyze
image_path = Path("output/ACCENTURE-Marker/_page_7_Figure_6.jpeg")

# Search queries for different aspects of the visualization
search_queries = [
    # Type identification
    "chart showing numeric trends",
    "graph with multiple data points",
    "statistical visualization",
    # Content queries
    "data showing growth over time",
    "comparison between different values",
    "percentage or ratio data",
    "market share information",
    "financial performance metrics",
    # Visual elements
    "contains data labels",
    "uses legend or key",
    "shows axis scales",
    "includes title or caption",
    # Data characteristics
    "shows clear pattern",
    "contains outlier points",
    "multiple data categories",
    "time series data",
]

# Find relevant content
results = find_relevant_content(image_path, search_queries)

# Print results
print("\n=== Content Search Results ===")
print("Showing relevance scores for different aspects of the visualization")
print("Higher scores indicate stronger presence of that aspect\n")

for query, score in results:
    relevance = score[0]
    if relevance > 0.5:  # Only show reasonably confident matches
        print(f"✓ {query}")
        print(f"  Relevance: {relevance:.3f}\n")

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


def identify_visualizations(image_directory, min_confidence=0.6):
    """Identify charts and infographics in extracted PDF images."""

    # Queries to identify different types of visualizations
    viz_queries = [
        "this is a chart or graph",
        "this is an infographic",
        "this shows data visualization",
        "this contains statistical information",
        "this is a business diagram",
    ]

    # Process queries once
    batch_queries = processor.process_queries(viz_queries).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)

    results = []
    total_images = 0

    # Process all images
    for img_path in Path(image_directory).glob("**/*.jpeg"):
        total_images += 1
        try:
            # Process image
            image = Image.open(img_path)
            batch_images = processor.process_images([image]).to(model.device)

            # Get embeddings and scores
            with torch.no_grad():
                image_embeddings = model(**batch_images)

            # Get max score across all viz queries
            scores = processor.score_multi_vector(query_embeddings, image_embeddings)
            max_score = max(score[0] for score in scores)

            if max_score > min_confidence:
                # Get the most relevant type
                viz_type = viz_queries[max(enumerate(scores), key=lambda x: x[1][0])[0]]
                results.append((img_path, viz_type, max_score))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Sort by confidence
    results.sort(key=lambda x: x[2], reverse=True)
    return results, total_images


# Directory containing extracted PDF images
image_dir = "output/ACCENTURE-Marker"

# Find visualizations
print(f"\nAnalyzing images in {image_dir}...")
visualizations, total = identify_visualizations(image_dir)

# Print results
print(f"\nFound {len(visualizations)} visualizations out of {total} total images")
print("\n=== Identified Visualizations ===")

for path, viz_type, confidence in visualizations:
    print(f"\n✓ {path.name}")
    print(f"  Type: {viz_type.replace('this is ', '').replace('this ', '')}")
    print(f"  Confidence: {confidence:.3f}")

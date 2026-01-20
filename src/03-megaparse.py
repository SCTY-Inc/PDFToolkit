from megaparse import MegaParse
from megaparse.parser.unstructured_parser import UnstructuredParser

# Initialize parser and MegaParse
parser = UnstructuredParser()
megaparse = MegaParse(parser)

# Parse PDF
response = megaparse.load("./docs/ACCENTURE_Luxury_2025.pdf")
print(response)

# Save to markdown
megaparse.save("./output/ACCENTURE-Megaparse.md")

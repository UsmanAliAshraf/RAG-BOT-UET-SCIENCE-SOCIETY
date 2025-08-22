from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re

# 1. Load markdown data
with open("data\data.md", "r", encoding="utf-8") as f:
    data = f.read()

# 2. Preprocess: Split by headings (H1, H2, H3)
# This keeps sections like "Collaborating Societies" and lists intact
pattern = r'(?:^|\n)(#+\s.*?)(?=\n#+|\Z)'
sections = re.findall(pattern, data, flags=re.DOTALL | re.MULTILINE)


# 3. Use free HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Build FAISS index
db = FAISS.from_texts(sections, embedding_model)

# 5. Save index locally
db.save_local("faiss_index")

print(f"✅ FAISS index built with {len(sections)} chunks, respecting headings and lists!")

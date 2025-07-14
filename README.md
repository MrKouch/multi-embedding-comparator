#Multi-Embedding Comparator

**Multi-Embedding Comparator** is a lightweight Python library , API and web interface that allows you to generate text
embeddings from different models and compare them using various distance metrics.

---

##Features

- Supports multiple sentence embedding models (e.g. `all-MiniLM-L6-v2`, `nli-roberta-base-v2`, OpenAI)
- Implements multiple distance metrics:
  - Cosine
  - Euclidean
  - Manhattan
  - Dot product
  - Angular
  - Chebyshev
  - Minkowski
  - Hamming
  - Jaccard
- Easily switch between models and metrics
- Simple Streamlit interface for interactive comparison
- Clean backend library structure for reuse

---

##Installation

Clone the repository and install it in **editable mode**:

```bash
git clone https://github.com/MrKouch/multi-embedding-comparator.git
cd multi-embedding-comparator
pip install -e .

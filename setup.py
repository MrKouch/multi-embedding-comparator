from setuptools import setup, find_packages

setup(
    name="embedding_distances",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[                                  # Required dependencies
        "torch>=1.13.0",
        "numpy>=1.21.0",
        "sentence-transformers>=2.2.0"
    ],
    author="Guy Kouchly and Altar Horowitz",
    author_email="altarhorowitz.email@example.com",
    description="A Python library for comparing distances between embedding vectors, "
                "with the option to choose embedding  and distance calculation metric type.",
    long_description=open("README.md").read(),          # Optional: pulls from README
    long_description_content_type="text/markdown",
    url="https://github.com/MrKouch/multi-embedding-comparator.git",  # Optional: GitHub link
    classifiers=[                                       # Metadata for PyPI (optional)
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

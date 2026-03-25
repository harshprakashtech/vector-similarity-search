# Vector Similarity Search

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)

A lightweight, efficient implementation of vector similarity search using **Cosine Similarity**. This project demonstrates the core concepts of embedding-based retrieval and nearest-neighbor search.

## Overview

This repository provides a foundational framework for searching across high-dimensional vector spaces. It is designed with a modular architecture, separating data management from the core search algorithms to allow for easy experimentation and scaling.

It serves as a foundational implementation for building more advanced systems such as semantic search, recommendation engines, and embedding-based retrieval pipelines.

## Key Features

- **Efficient Computation**: Leverages NumPy for high-performance vector operations.
- **Cosine Similarity**: Uses the standard metric for measuring directional similarity between vectors.
- **Modular Design**: Clear separation between `src/core` (logic), `src/data` (datasets) and `scripts` (execution).
- **Extensible**: Easily swap out similarity metrics or update the underlying dataset.

## Project Structure

```text
.
├── src/
│   ├── core/           # Core search algorithms and logic
│   │   └── vector_search.py
│   └── data/           # Data loading and vector storage
│       └── loader.py
├── scripts/            # Example scripts and utilities
│   └── main.py
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Technologies Used

- **Python**: Core programming language.
- **NumPy**: For efficient multi-dimensional array operations and linear algebra.

---

_Created as part of a learning journey into Machine Learning and NLP._

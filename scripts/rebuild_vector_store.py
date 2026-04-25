from __future__ import annotations

import argparse
from pathlib import Path

from utils.file_handler import clear_vector_store, rebuild_vector_store_from_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild DocuMind FAISS vector store.")
    parser.add_argument(
        "--vector-store-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "vector_db"),
        help="Path to vector store directory (default: data/vector_db).",
    )
    parser.add_argument(
        "--embed-model-name",
        default="all-MiniLM-L6-v2",
        help="Embedding model name to use for rebuild.",
    )
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Delete vector store only (no rebuild).",
    )
    args = parser.parse_args()

    vector_store_dir = Path(args.vector_store_dir)
    if args.clear_only:
        clear_vector_store(vector_store_dir)
        print(f"Cleared vector store: {vector_store_dir}")
        return

    rebuilt = rebuild_vector_store_from_texts(
        vector_store_dir=vector_store_dir,
        embed_model_name=args.embed_model_name,
    )
    print(f"Rebuilt vector store with {rebuilt} chunk(s): {vector_store_dir}")


if __name__ == "__main__":
    main()

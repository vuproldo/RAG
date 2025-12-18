import check_env
check_env.main()

from embedding import create_embeddings
from build_faiss import build_index
from rag_module import answer_question
import os

FAISS_PATH = os.path.join("rag", "vector_index.bin")


def main():
    if not os.path.exists(FAISS_PATH):
        create_embeddings()
        build_index()

    print("\n=== RAG Trainer Chat ===")
    while True:
        q = input("\nВопрос: ")
        if q.lower() in ("exit", "quit"):
            break
        print("\nОтвет:\n")
        print(answer_question(q))


if __name__ == "__main__":
    main()

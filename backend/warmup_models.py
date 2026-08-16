#!/usr/bin/env python3
import bootstrap
from reranker import MODEL_NAME, Reranker

def main() -> None:
    print(f'==> Caching reranker model ({MODEL_NAME})')
    Reranker()
    print('==> Local models ready')

if __name__ == '__main__':
    main()

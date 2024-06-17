#!/usr/bin/env python
import sys
import chromadb
from package import *

def main():
    ask_user_to_create_db()
    while True:
        prepare_db()
        iter_num = ask_how_many_documents()
        responses = response_loop(iter_num)
        ask_if_more_to_ask()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
#!/usr/bin/env python3
import argparse
import json
import os
import sys

# Ensure project root is on sys.path when running as a script
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from index.chroma_store import ChromaStore
from index.ingest import load_and_chunk
from graph.builder import GraphPipeline


def cmd_ingest(args):
    ids, texts, metas = load_and_chunk(args.paths, tags=args.tags)
    store = ChromaStore()
    added = store.add_chunks(ids, texts, metas)
    print(json.dumps({"added": added, "collection_size": store.count()}, indent=2))


def cmd_query(args):
    store = ChromaStore()
    pipeline = GraphPipeline(store)
    out = pipeline.run(args.query, top_k=args.k)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()

    p_ing = p.add_argument_group("ingest")
    p_sub_ing = sub.add_parser("ingest")
    p_sub_ing.add_argument("paths", nargs="+", help="Files or directories to ingest")
    p_sub_ing.add_argument("--tags", nargs="*", default=[], help="Tags to attach")
    p_sub_ing.set_defaults(func=cmd_ingest)

    p_q = p.add_argument_group("query")
    p_sub_q = sub.add_parser("query")
    p_sub_q.add_argument("query", help="User question")
    p_sub_q.add_argument("-k", type=int, default=5)
    p_sub_q.set_defaults(func=cmd_query)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

"""Build the fixed 1200-character / 200-overlap control corpus."""

import argparse
import json

from stage2.data.fixed_chunker import build_fixed_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fixed_chunk.json")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    count = build_fixed_chunks(
        config["papers_path"],
        config["retrieval_chunks_path"],
        config["chunk_max_chars"],
        config["chunk_overlap_chars"],
    )
    print(f"Wrote {count} fixed control chunks -> {config['retrieval_chunks_path']}")


if __name__ == "__main__":
    main()

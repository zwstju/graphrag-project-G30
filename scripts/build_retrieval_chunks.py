import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_retrieval_chunks(source_path: Path, output_path: Path) -> dict:
    source_hash_before = sha256_file(source_path)
    source_count = 0
    excluded_references = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            with source_path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON at {source_path}:{line_number}: {exc}"
                        ) from exc

                    source_count += 1
                    if chunk.get("section") == "References":
                        excluded_references += 1
                        continue
                    output.write(line if line.endswith("\n") else line + "\n")

        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    output_count = 0
    remaining_references = 0
    with output_path.open("r", encoding="utf-8") as output:
        for line_number, line in enumerate(output, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {output_path}:{line_number}: {exc}"
                ) from exc
            output_count += 1
            if chunk.get("section") == "References":
                remaining_references += 1

    source_hash_after = sha256_file(source_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError(f"Source corpus changed while filtering: {source_path}")
    if output_count != source_count - excluded_references:
        raise RuntimeError(
            "Retrieval corpus count mismatch: "
            f"{output_count} != {source_count} - {excluded_references}"
        )
    if remaining_references:
        raise RuntimeError(
            f"Retrieval corpus still contains {remaining_references} References chunks"
        )

    return {
        "source_count": source_count,
        "excluded_references": excluded_references,
        "output_count": output_count,
        "remaining_references": remaining_references,
        "source_sha256": source_hash_before,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exclude canonical References chunks from the Stage 2 retrieval corpus."
    )
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)

    source_path = Path(config["chunks_path"])
    output_path = Path(config["retrieval_chunks_path"])
    stats = build_retrieval_chunks(source_path, output_path)

    print(f"Source corpus: {source_path}")
    print(f"Retrieval corpus: {output_path}")
    print(f"Source chunk count: {stats['source_count']}")
    print(f"Excluded References: {stats['excluded_references']}")
    print(f"Output chunk count: {stats['output_count']}")
    print(f"Remaining References: {stats['remaining_references']}")
    print(f"Source SHA-256: {stats['source_sha256']}")


if __name__ == "__main__":
    main()

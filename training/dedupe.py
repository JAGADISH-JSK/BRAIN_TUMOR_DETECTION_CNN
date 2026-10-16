"""Remove EXACT duplicate images (same bytes) from a folder tree.

Usage: python -m training.dedupe data/Training
Note: this only catches byte-identical files. Near-duplicates (resized/re-saved copies)
need perceptual hashing, e.g. the `imagehash` package. Always check your split for leakage.
"""
import hashlib
import os
import sys


def main(root: str) -> None:
    seen, removed = {}, 0
    for folder, _, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(folder, name)
            with open(path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
            if digest in seen:
                os.remove(path)
                removed += 1
            else:
                seen[digest] = path
    print(f"Removed {removed} exact duplicates from {root}")


if __name__ == "__main__":
    main(sys.argv[1])

"""Duplicate detection: exact (content hash) and near (MinHash + LSH banding).

Exact: two records whose normalised text hashes to the same SHA-256 are the same
document uploaded twice. We keep the first and point the second at it.

Near: "same text with small edits", measured as Jaccard similarity of the sets of
3-word shingles: |A ∩ B| / |A ∪ B|. We call two records near-duplicates when
Jaccard >= 0.8.

Computing Jaccard for every pair of 62k records is ~2 billion comparisons, so:
  1. MinHash: apply 64 different hash functions to every shingle and keep the
     minimum of each. For one hash function, P(minA == minB) equals Jaccard(A, B),
     so the 64 minimums are a compact signature of the shingle set.
  2. LSH banding: cut the signature into 16 bands of 4 values. Two records become
     candidates if any band is identical. With Jaccard 0.84 the chance of at least
     one identical band is 1 - (1 - 0.84^4)^16 ≈ 0.99999; with Jaccard 0.3 it is
     about 0.12. So true near-duplicates are almost always compared, most
     unrelated pairs never are.
  3. Verify each candidate with the exact Jaccard; only >= 0.8 counts.

Why not SimHash (tried first): on 35-40 word texts that shared 84% of shingles,
64-bit SimHash fingerprints still differed in 8-10 bits, far beyond the usual
<= 3-bit cut-off, which is tuned for long web pages. Measured, then replaced.
"""
import hashlib
import re
from collections import defaultdict

import numpy as np

N_HASHES = 64
BANDS = 16
ROWS = N_HASHES // BANDS
THRESHOLD = 0.8
_PRIME = (1 << 61) - 1
_rng = np.random.default_rng(0)                       # fixed: signatures must be reproducible
_A = _rng.integers(1, _PRIME, N_HASHES, dtype=np.uint64)
_B = _rng.integers(0, _PRIME, N_HASHES, dtype=np.uint64)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 3) -> set:
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def minhash(sh: set) -> np.ndarray:
    """64 minimums of (a*x + b) mod p over the shingle hashes x."""
    if not sh:
        return np.zeros(N_HASHES, dtype=np.uint64)
    x = np.fromiter((int.from_bytes(hashlib.blake2b(s.encode(), digest_size=7).digest(), "big")
                     for s in sh), dtype=np.uint64, count=len(sh))
    # uint64 arithmetic wraps instead of overflowing; good enough to spread values
    return ((x[:, None] * _A + _B) % np.uint64(_PRIME)).min(axis=0)


def lsh_keys(sig: np.ndarray):
    """[(band, key)] with each band of 4 minimums hashed to one signed 64-bit int (fits a Postgres bigint)."""
    return [(b, int.from_bytes(hashlib.blake2b(sig[b * ROWS:(b + 1) * ROWS].tobytes(), digest_size=8).digest(),
                               "big", signed=True)) for b in range(BANDS)]


def find_duplicates(records):
    """Return {doc_id: (kept_doc_id, "exact" | "near", similarity)} for every duplicate.

    The first record seen is kept; later copies point at it.
    """
    dupes = {}
    seen_hash = {}
    kept_shingles = {}
    buckets = defaultdict(list)                       # (band, band values) -> [kept doc_id]
    for rec in records:
        h = content_hash(rec.text)
        if h in seen_hash:
            dupes[rec.doc_id] = (seen_hash[h], "exact", 1.0)
            continue
        seen_hash[h] = rec.doc_id
        sh = shingles(rec.text)
        sig = minhash(sh)
        keys = lsh_keys(sig)
        candidates = {other for key in keys for other in buckets[key]}
        best = max(((jaccard(sh, kept_shingles[o]), o) for o in candidates), default=(0.0, None))
        if best[0] >= THRESHOLD:
            dupes[rec.doc_id] = (best[1], "near", round(best[0], 3))
            continue
        kept_shingles[rec.doc_id] = sh
        for key in keys:
            buckets[key].append(rec.doc_id)
    return dupes

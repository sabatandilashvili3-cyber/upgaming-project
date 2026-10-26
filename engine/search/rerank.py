"""Second-stage re-ranking with a cross-encoder.

A bi-encoder (our embedder) turns the question and each passage into vectors separately,
then compares them; that is what makes searching 63k chunks fast. A cross-encoder reads
the question and one passage TOGETHER and outputs a relevance score, so it can see that
"which change caused INC-2026-004" and a postmortem's "Root cause: CHG-2458" belong
together. It is far too slow for 63k passages, so it only re-orders the top N (30) that
hybrid search already found: cheap recall first, expensive precision second.

Model: cross-encoder/ms-marco-MiniLM-L-4-v2 (trained on Bing search queries, CPU-friendly). The
re-ranker is ~93% of a re-ranked search's time on the 4-core ARM server, so the model size is the
throughput lever. Measured on all 539 retrieval questions (eval/retrieval.py, H-routed-15):
  L-6: 345 ms per re-rank, recall@10 0.872, MRR 0.915
  L-4: 233 ms,             recall@10 0.871, MRR 0.911   <- 1.5x faster for 0.001 recall
  L-2: 120 ms,             recall@10 0.862, MRR 0.902
int8 dynamic quantisation was tried and was slower on ARM (qnnpack): 569 ms vs 347 ms.
"""
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-4-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, max_length: int = 256):
        # 256 tokens covers a 200-word chunk; 512 doubled the cost on the server's CPU for no
        # measured gain (load test: 30 pairs x 512 tokens took 0.7-1.7 s per question)
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank_fused(self, query: str, hits, top_n: int = 30):
        """Fuse the cross-encoder's order with the incoming order (RRF) instead of replacing it.

        Measured: replacing the order lifted HotpotQA but dropped Meridian recall@10 from 0.78 to
        0.44, because the cross-encoder (trained on web queries) cannot see that CHG-2458 or
        "March 2026" matter, and its order discarded the id/date boost and per-file diversity.
        Fusing keeps both signals: a passage must look relevant to the cross-encoder AND to
        the investigation-aware ranking to rise.
        """
        from .fuse import rrf
        head, tail = hits[:top_n], hits[top_n:]
        if not head:
            return hits
        scores = self.model.predict([(query, h.text) for h in head], batch_size=32, show_progress_bar=False)
        ce_order = [h.chunk_id for _, h in sorted(zip(scores, head), key=lambda x: -x[0])]
        by_id = {h.chunk_id: h for h in head}
        fused = rrf([[h.chunk_id for h in head], ce_order])
        return [by_id[cid] for cid, _ in fused] + tail

    def rerank(self, query: str, hits, top_n: int = 30):
        """Re-order the first top_n hits by cross-encoder score; the rest keep their order after them."""
        head, tail = hits[:top_n], hits[top_n:]
        if not head:
            return hits
        scores = self.model.predict([(query, h.text) for h in head], batch_size=32, show_progress_bar=False)
        for h, s in zip(head, scores):
            h.score = float(s)
        return sorted(head, key=lambda h: h.score, reverse=True) + tail

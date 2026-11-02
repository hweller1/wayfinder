"""Demo lane: a composition root and CLI (`scout-demo`) that runs scout against REAL turbopuffer, Voyage and
an OpenAI-compatible chat endpoint, with the real in-process `GatewayService` so the secure-indexing path
(encrypted paths and line ranges, HMAC tokens, Merkle sync) is exercised exactly as in the product.

This package sits above `scout.surfaces` in the import layers: it may import every lower layer and nothing
imports it. It is a demo harness, not the product wiring (`scout.wiring` is).
"""

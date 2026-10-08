# Final benchmark handoff checklist

Each box requires recorded evidence; a checked box is not implied by this template.

## Before transferring to the future machine

- [ ] Step 10 development-dataset annotation decision and freeze manifest approved.
- [ ] Step 10 300-variant full-generation PILOT complete, or its precise memory-safety stop recorded.
- [ ] Pilot checkpoint/resume, unique IDs, leakage audit, unsafe-return audit and blank human review sample verified.
- [ ] Current Git commit, dirty status, code fingerprint, dataset/corpus/index hashes and two Qwen shard hashes recorded.
- [ ] Python and installed dependency versions recorded; no dependency change made merely to create this snapshot.

## Future corpus and dataset

- [ ] Actual hardware, RAM, pagefile, GPU/VRAM, OS and runtime profile recorded.
- [ ] Approved source inventory populated; unknown dates/URLs remain blank.
- [ ] Conflicting or superseded official versions explicitly adjudicated.
- [ ] PDFs discovered recursively; duplicate/bad/partial/empty files reviewed.
- [ ] Stage A (5–10 PDFs) and Stage B (20–30 PDFs) pass their ingestion/index/memory checks.
- [ ] Full corpus ingested; suspicious chunks inspected; fresh dense/sparse indexes verified.
- [ ] Independent final dataset validated, annotation version frozen, overlap with development set disclosed.
- [ ] Dataset SHA-256, source inventory, corpus fingerprint, index fingerprint and model SHA-256 values match copied artifacts.
- [ ] Final-corpus 100–300-question pilot checks schema, index, memory and latency without tuning production answers.

## Final run

- [ ] Fixed strict full-generation `FINAL_BENCHMARK` configuration approved; answer bank and reranker OFF.
- [ ] Git/code, corpus/index, dataset, model and configuration lock captured in the run manifest.
- [ ] Sequential inference with atomic checkpoints and resume enabled.
- [ ] No code, source, annotation, model or setting changes during the run; invalidate/restart if required.
- [ ] Expected IDs equal completed IDs; no duplicate IDs; errors have explicit `SYSTEM_ERROR` rows.
- [ ] Retrieval, answerability, safety, latency, memory, parity and automatic-proxy tables produced with denominators.
- [ ] Human-review sample stratified and reviewers assigned; genuine scores imported separately.
- [ ] Final report distinguishes automatic proxies, SafeAnswerCoverage and human-rated correctness.
- [ ] Complete reproducibility package copied and hashes verified after transfer.

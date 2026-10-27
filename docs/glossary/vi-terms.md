# Vietnamese Glossary — Single Source of Truth for Technical Terms

- Agent: docs-b0-gloss
- Date: 2026-10-26
- Scope: LiteTorch docs rewrite, Batch B0
- Source of seed: `workspace/framework/docs/plan/verification.md` §5.2 (30 terms)
- Cross-checked against: `workspace/framework/integration/docs-vie/*.md` (12 files, 1,736 lines, tree HEAD 88a4ba1)
- Status: living document. New terms are proposed as a patch to THIS file and approved by the Docs leader before use. No inline invention.

## 1. Rules

1. **Code is never translated.** Code identifiers, API names, class names, function names, error strings, file paths, command lines, dtype/device enum values, and math symbols stay exactly as in the English original. Only prose and `print()` string literals are translated.
2. **One EN term = one VI rendering, everywhere.** Two different VI renderings of one EN term in one file is a FAIL on the glossary lint (see §4).
3. **Keep-EN is a decision, not laziness.** A term marked "keep EN" is kept in English in VI prose (e.g. `tensor`, `loss`, `optimizer`) because that is the established usage in the existing VI docs and in the Vietnamese ML community. Do not "fix" a keep-EN term into Vietnamese on your own initiative.
4. **Translate meaning, not words.** Examples keep identical code, identical numbers, identical structure in both languages. A VI sentence that changes the technical meaning is a defect even if the words look translated.
5. **First use defines.** The first time a translated term appears on a page, it may carry the English original in parentheses, e.g. "lượng tử hóa (quantization)". After that, use the VI term alone.
6. **Headings follow the table.** A heading that uses a non-glossary rendering (e.g. "Hàm Mất Mát" for "loss function") is a defect — see §5, I-3.
7. **No new terms without approval.** If a term is missing from this table, propose it here first (EN term, proposed VI term, evidence of usage or a rationale). Do not invent a translation inline in a page.

## 2. Do-not-translate list (never rendered in Vietnamese)

- All code identifiers: `Tensor`, `Module`, `Linear`, `Ops.matmul`, `lt.cuda`, `requires_grad`, `state_dict`, `DataLoader`, etc.
- All API, class, function, and enum names, including `DataType.FP32`, `Device("cuda")`, `litetorch.Device`.
- All error strings and log lines (quote verbatim, then explain in Vietnamese around them).
- File paths, URLs, command lines, shell output, version numbers.
- Single-letter and Greek math symbols (e.g. `x`, `W`, `ε`, `α`).
- Acronyms kept as-is: AMP, DAG, JIT, QAT, PTQ, NCCL, cuDNN, cuBLAS, ROCm, TPU, LLM, RAG (when it appears), FP8/INT8/FP16/BF16, API, CPU, GPU.

## 3. Term table

"VI term" = the single approved Vietnamese rendering. "Keep EN" means the English word itself is used in VI prose. "Evidence" = occurrence counts in the existing 12 VI sections (docs-vie/, HEAD 88a4ba1); counts are case-insensitive prose+code matches unless noted.

### 3.1 Seed terms (from verification.md §5.2, corrected against actual usage — see §5)

| EN term | VI term | Notes / evidence |
|---|---|---|
| tensor | tensor (keep EN) | Pervasive, never translated. |
| autograd / automatic differentiation | đạo hàm tự động (Autograd) | Title of §4: "Đạo Hàm Tự Động (Autograd)". "Vi phân" is NOT the term for autograd — see I-1. |
| gradient | gradient (keep EN) | 6 occurrences in VI prose kept EN ("Lan truyen gradient", "Xóa sạch gradient"). "Đạo hàm" is the derivative (the mathematical concept), NOT a translation of "gradient". Never write "gradient" as "đạo hàm" and never write a derivative as "gradient". |
| derivative | đạo hàm | The mathematical concept. Distinct from "gradient" above. |
| backward pass | backward pass (keep EN) | SEED CORRECTED: seed proposed "lượt ngược", but existing VI uses "forward pass"/"backward pass" untranslated (0 hits for "lượt" in docs-vie/). Adopted existing usage — see I-8. |
| forward pass | forward pass (keep EN) | Same as backward pass — see I-8. |
| loss | loss (keep EN) | 62 occurrences kept EN. Headings must use "hàm loss", not "hàm mất mát" — see I-3. |
| optimizer | optimizer (keep EN) | SEED CORRECTED: seed proposed "bộ tối ưu"; actual usage is "optimizer" 36x vs "bộ tối ưu hóa" 2x. "Bộ tối ưu" itself occurs 0x. Adopted keep-EN — see I-7. |
| learning rate | tốc độ học | 3x vs "learning rate" 1x — see I-10. |
| batch | batch (keep EN) | 24x. "Batch size" stays "batch size". |
| epoch | epoch (keep EN) | 4x. |
| device | thiết bị | 12x. |
| dtype / data type | kiểu dữ liệu | 4x. |
| shape | shape (keep EN) | 20x. |
| stride | stride (keep EN) | 18x. |
| memory eviction | giải phóng bộ nhớ | NEW term, no prior usage (0 hits for "eviction"/"giải phóng" in docs-vie/). LiteTorch differentiator — define once on first use, reuse everywhere. |
| caching allocator | caching allocator (keep EN) | SEED CORRECTED: seed proposed "bộ cấp phát đệm"; actual usage is "allocator" 5x vs "bộ cấp phát" 1x. Adopted keep-EN — see I-9. |
| kernel | kernel (keep EN) | 11x vs "hạt nhân" 3x — see I-2. |
| fusion (JIT) | fusion (keep EN) | "fused" 15x, "fusion" 1x vs "hợp nhất" 1x — see I-5. "Fused kernel" stays "fused kernel". |
| quantization | lượng tử hóa | 13x, consistent. Headings: "Lượng Tử Hóa (Quantization)". |
| distributed training | huấn luyện phân tán | 1x. |
| data parallel | song song dữ liệu | No prior usage — new term, seed stands. |
| model parallel | song song mô hình | No prior usage — new term, seed stands. |
| pipeline parallel | song song pipeline | No prior usage — new term, seed stands. |
| checkpoint | checkpoint (keep EN) | 7x. |
| mixed precision (AMP) | độ chính xác hỗn hợp (AMP) | 2x "độ chính xác". Keep the AMP acronym. |
| embedding | embedding (keep EN) | 7x. |
| attention | attention (keep EN) | 24x. |
| tokenizer | tokenizer (keep EN) | 0 hits in docs-vie/ (tokenizers.md is a separate bilingual file); keep EN per convention. |
| inference | suy luận | 3x; "inference" 0x in VI prose. |
| throughput | throughput (keep EN) | 1x bare vs "thông lượng (throughput)" 1x — see I-11. |

### 3.2 Expanded terms (recurring in docs-vie/, verified by scan)

| EN term | VI term | Notes / evidence |
|---|---|---|
| weight | trọng số | 11x, consistent. |
| bias | bias (keep EN) | 24x. |
| layer | lớp | 10x ("lớp "). |
| model | mô hình | 16x. |
| parameter | tham số | 5x. |
| dataset | tập dữ liệu | 3x. The class name `Dataset` stays EN. |
| dataloader / DataLoader | DataLoader (keep EN) | 13x. |
| scheduler | scheduler (keep EN) | 13x; "bộ lập lịch" 0x. |
| activation | activation (keep EN) | 6x vs "hàm kích hoạt" 1x — see I-6. |
| normalization | chuẩn hóa | 5x. |
| graph (DAG) | đồ thị | 6x ("đồ thị có hướng không chu trình (DAG)"). |
| loss function | hàm loss | Fixes "hàm mất mát" — see I-3. |
| training | huấn luyện | Pervasive ("quá trình huấn luyện", "Huấn luyện FP8"). |
| underflow | tràn số dưới | CORRECTS the existing rendering "mất mát số học (underflow)" — see I-4. |
| overflow | tràn số trên | New term, no prior usage; pair of "underflow". |
| latency | độ trễ | 9.llm_serving.md:3, "giảm độ trễ (latency)". |
| quantization-aware training (QAT) | huấn luyện nhận biết lượng tử (QAT) | 10.quantization_and_jit.md:3. |
| post-training quantization (PTQ) | lượng tử hóa sau huấn luyện (PTQ) | 10.quantization_and_jit.md:3. |
| fake quantization | fake quantization (keep EN) | 10.quantization_and_jit.md:9, "(Fake Quantization)". |
| weight decay | weight decay (keep EN) | 7.optimizers_and_amp.md:29, "(weight decay)". |
| recompute | tính toán lại | B4-expl (E3): approved by the leader 2026-10-27. |
| migration (guide) | chuyển đổi | B4-front2: used consistently in migration.md ("Hướng dẫn chuyển đổi"). |
| behavioral difference | khác biệt hành vi | B4-front2: used consistently in behavioral-differences.md. |
| troubleshooting | khắc phục sự cố | B4-front2: used consistently in troubleshooting.md. |
| changelog | nhật ký thay đổi | B4-front2: used consistently in changelog.md. |
| silent fallback | rơi về lặng lẽ | B4-front2: "to(device) rơi về CPU lặng lẽ" (D-02). |
| workaround | cách khắc phục tạm thời | B4-front2: used consistently across migration/troubleshooting/differences pages. |

## 4. Consistency enforcement

1. The extractor (`docs/tools/extract_examples.py`, Batch B0 harness) runs a glossary lint over every VI page:
   - Two different VI renderings of one EN term in one file = FAIL.
   - An EN term that has a VI mapping in §3 appearing untranslated in VI prose where the mapping applies = FAIL (keep-EN terms are exempt by definition).
   - A VI term from the "not approved" set in §5 appearing in a page = FAIL (the page must use the approved rendering).
2. New terms are proposed by section authors as a patch to this file and approved by the Docs leader before use; no inline invention (rule 7, §1).
3. A VI reviewer (one worker per batch, rotating) reads the VI pages end-to-end for naturalness; the glossary lint catches consistency, humans catch meaning.

## 5. Open inconsistencies in the existing VI (to be closed by the rewrite)

Each item: what the existing VI does, what this glossary decides, and where the evidence is. Section authors must apply the decision when rewriting the cited page.

- **I-1 — "vi phân" vs "đạo hàm" for differentiation.** §4 uses "đạo hàm tự động" throughout ("Đạo Hàm Tự Động (Autograd)", "đồ thị đạo hàm tự động"), but 10.quantization_and_jit.md:131 writes "đồ thị vi phân giải tích (Analytical Backward Graph)". Decision: "đạo hàm" family only. "Vi phân" is not a glossary term and must not appear as a translation of autograd/differentiation. Fix in the §10 rewrite.
- **I-2 — kernel: "kernel" vs "hạt nhân".** "kernel" 11x; "hạt nhân" 3x — 1.index.md:54 ("các hạt nhân (kernels) LLM", EN glossed) and 5.operators.md:3 ("các hạt nhân LLM tối ưu hóa", bare). Decision: keep EN "kernel". Rewrite replaces both "hạt nhân" occurrences.
- **I-3 — loss function: "hàm loss" vs "hàm mất mát".** "loss" kept EN 62x, but two headings use "Hàm Mất Mát": 5.operators.md:105 ("Hàm Mất Mát (Loss Functions)") and 6.neural_networks.md:142 ("Các Module Hàm Mất Mát (Loss Modules)"). Decision: headings become "Hàm Loss" / "Các Module Hàm Loss". "Mất mát" must not be used for "loss" anywhere.
- **I-4 — underflow mistranslated.** 7.optimizers_and_amp.md:79 renders underflow as "mất mát số học (underflow)". That reads as "arithmetic loss" and is wrong. Decision: "tràn số dưới" (§3.2). Fix in the §7 rewrite.
- **I-5 — fusion: "fusion" vs "hợp nhất".** "fused" 15x, "fusion" 1x; "hợp nhất" 1x at 10.quantization_and_jit.md:76 ("động cơ hợp nhất Kernel JIT (Kernel Fusion Engine)"). Decision: keep EN "fusion"/"fused".
- **I-6 — activation: "activation" vs "hàm kích hoạt".** "activation" 6x vs "hàm kích hoạt" 1x (single occurrence; exact line not pinned in this pass — re-verify during B1). Decision: keep EN "activation".
- **I-7 — optimizer: "optimizer" vs "bộ tối ưu hóa".** "optimizer" 36x vs "bộ tối ưu hóa" 2x (7.optimizers_and_amp.md:9, 8.distributed_training.md:49). The seed's "bộ tối ưu" occurs 0x — it matches neither rendering. Decision: keep EN "optimizer". Seed corrected.
- **I-8 — forward/backward pass: kept EN vs seed "lượt xuôi/lượt ngược".** Existing VI uses "forward pass"/"backward pass" untranslated (e.g. 4.tensor_and_autograd.md:125, 10.quantization_and_jit.md:9); "lượt" occurs 0x in docs-vie/. Decision: keep EN. Seed corrected.
- **I-9 — allocator: "allocator" vs "bộ cấp phát".** "allocator" 5x vs "bộ cấp phát" 1x (single occurrence; exact line not pinned in this pass — re-verify during B1). Decision: keep EN "caching allocator". Seed corrected.
- **I-10 — learning rate: "tốc độ học" vs "learning rate".** "tốc độ học" 3x vs "learning rate" 1x. Decision: "tốc độ học" (seed stands; the single EN occurrence is fixed in the rewrite).
- **I-11 — throughput: "throughput" vs "thông lượng".** 1x each; 9.llm_serving.md:3 uses "thông lượng (throughput)" with EN gloss. Decision: keep EN "throughput" bare.

## 6. Seed deviations summary (verification.md §5.2 vs this file)

Four seed terms were changed on evidence from the existing VI (counts above):

| Seed term | Seed said | This glossary decides | Why |
|---|---|---|---|
| forward pass | lượt xuôi | forward pass (keep EN) | 0 hits for "lượt"; existing VI keeps EN |
| backward pass | lượt ngược | backward pass (keep EN) | Same as forward pass |
| optimizer | bộ tối ưu | optimizer (keep EN) | 36x EN vs 2x "bộ tối ưu hóa" vs 0x "bộ tối ưu" |
| caching allocator | bộ cấp phát đệm | caching allocator (keep EN) | "allocator" 5x vs "bộ cấp phát" 1x |

One seed rule was clarified, not changed: "gradient" is keep-EN; "đạo hàm" is the derivative (a different concept), not a translation of "gradient" — the two must never be swapped.

## 7. Changelog

- 2026-10-26 (docs-b0-gloss, B0): created. 50 terms (30 seed + 20 expanded). 11 open inconsistencies logged (I-1–I-11), 4 seed deviations decided on evidence.
- 2026-10-27 (docs-leader, B4): approved + added 6 terms proposed by docs-b4-front2 (chuyển đổi, khác biệt hành vi, khắc phục sự cố, nhật ký thay đổi, rơi về lặng lẽ, cách khắc phục tạm thời). Total 56 terms.
- 2026-10-27 (docs-leader, B4): approved + added 1 term proposed by docs-b4-expl (recompute → tính toán lại). Total 57 terms. (swap-in/out left unglossaried — no rendering proposed; keep EN until proposed with evidence.)

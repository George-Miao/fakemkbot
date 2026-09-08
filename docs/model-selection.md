# Model selection for mandatory substring inclusion

Research date: 2026-09-01. This report applies to `transformers==4.56.2` and the current RTX 3070 system.

## Decision

Keep **`Qwen/Qwen2.5-1.5B-Instruct`** as the production model. A local UMT5-base trial proved that dual-sentinel output can enforce exact inclusion, but its generated joins were less fluent than the current Qwen output.

Use **`Qwen/Qwen3-1.7B` in non-thinking mode** as the next quality experiment. It is post-trained for creative writing, dialogue, and instruction following. Do not promote UMT5-base without a larger corpus or a blind evaluation that shows a clear fluency gain.

Do not ask any model to guarantee inclusion by prompt alone. Make exact inclusion an application invariant:

```python
required_text in final_reply
```

Preserve the original Python string, let the model generate only its surrounding text or a placeholder, splice in the original string, and validate again in Python and Rust before delivery. This gives exact Unicode inclusion at any position while normal sampling remains enabled.

## Requirement

For each request string `required_text`, every delivered result must contain the same Unicode code-point sequence. A paraphrase, normalized equivalent, token match in the prompt, or related concept does not pass.

The current pipeline has these constraints:

- RTX 3070 with 8,192 MiB VRAM
- About 1,176 short Chinese training messages
- Python 3.12, Torch 2.8.0, Transformers 4.56.2, PEFT 0.17.1, and bitsandbytes 0.50.2
- NF4 double-quantized QLoRA with rank 16, sequence length 256, batch size 1, and gradient checkpointing
- One persistent Python generation worker started by the Rust bot
- Sampling with temperature 0.9 and top-p 0.9

The current Python worker strips the request and output. Exact leading or trailing whitespace would therefore be lost. The new protocol must carry an untouched `required_text` field. A cleaned copy can be used in the model prompt, but it cannot replace the original value.

## Recommended trial order

The ratings below are engineering judgments. Official benchmarks do not measure fidelity to this private style corpus. A controlled local comparison is required.

| Rank | Model | Generation design | Main benefit | RTX 3070 fit | Decision |
|---:|---|---|---|---|---|
| 1 | UMT5-base | Generate left and right missing spans | Direct task fit and newer multilingual pretraining | Measured at 2,507 MiB allocated | Exact inclusion passed, fluency failed |
| 2 | Qwen3-1.7B | Generate one marker, then splice | Strong multilingual instruction and creative chat capability | Good with NF4 QLoRA | Likely naturalness challenger |
| 3 | Qwen2.5-1.5B-Instruct | Generate one marker, then splice | Existing adapter and proven local operation | Proven | Quality baseline |
| 4 | ByT5-small | Generate left and right missing spans | Raw-byte conditioning for noisy text | Good with LoRA | Noise ablation |
| 5 | Retrieval or fixed templates | Insert the raw text into a selected pattern | Exact, fast, and no training | Excellent | Deterministic baseline |
| 6 | Character n-gram or recurrent LM | Generate forward and backward from the anchor | Very small and easy to train | Excellent | Memorization diagnostic |
| 7 | Insertion Transformer or GLM | Insert or fill around a fixed anchor | Architecture matches flexible generation | Model and tooling gap | Research only |

## Local UMT5 trial

The CUDA trial used `google/umt5-base`, NF4 loading, rank-16 LoRA on `q`, `k`, `v`, `o`, `wi_0`, `wi_1`, and `wo`, and two epochs. It trained 6,782,976 parameters on 3,241 anchored records and evaluated on 176 anchored records.

The final evaluation loss was 2.9733. All 59 seeded held-out generations had valid sentinel structure, all included the untouched anchor after splicing, and none repeated the anchor. Peak CUDA memory was 2,507 MiB allocated and 2,548 MiB reserved.

The structure result passed, but the language result failed. Custom outputs included `今天我是好了`, `要cuda有多个`, `绷不住了都可以`, `不RTX 3070是`, and `你🤔`. The anchor placement was exact, but most joins were incomplete or ungrammatical. Held-out samples had the same problem. The current Qwen adapter produces more complete colloquial Chinese, so the UMT5 adapter was not deployed.

This was not a blind model comparison because the current Qwen worker does not yet implement the same marker-and-splice contract. The result is sufficient to reject UMT5-base as the next production default, not to rank all later models.

### UMT5-base

UMT5-base is the best architecture-first candidate. Its official card lists Chinese among 107 languages and says it uses a refreshed 29 trillion character mC4 corpus with UniMax language sampling. It is pretraining-only and needs task fine-tuning. Transformers 4.56.2 supports its conditional-generation architecture and shows sentinel infilling.

Sources:

- [Transformers 4.56.2 UMT5 documentation and infilling example](https://github.com/huggingface/transformers/blob/v4.56.2/docs/source/en/model_doc/umt5.md#L24-L79)
- [UMT5-base official model card](https://huggingface.co/google/umt5-base)
- [UMT5-base architecture config](https://huggingface.co/google/umt5-base/blob/main/config.json)
- [UniMax paper](https://openreview.net/forum?id=kXwdL1cWOAi)

The training format maps directly to this product:

```text
input:  <extra_id_0>required text<extra_id_1>
target: <extra_id_0>left context<extra_id_1>right context<extra_id_2>
final:  left context + original required_text + right context
```

The model never generates the protected text. The application parses the two generated spans and places the untouched Python string between them. An empty left or right span naturally places the text at the start or end. Reject malformed sentinel output and validate the final string.

This method also makes efficient use of the small corpus. Each source message can produce several supervised examples with different contiguous anchor spans. Split by source message before this expansion so related examples do not cross the validation boundary.

The main risk is boundary quality. A base model with only 1,176 source messages can generate a left span that leads into one meaning and a right span that follows another. It can also repeat or paraphrase the anchor in its generated spans. Exact inclusion can pass while the result sounds wrong. Blind review must measure natural joins.

### ByT5-small

ByT5 is the byte-level ablation. It reads raw UTF-8 bytes, uses an encoder-decoder Transformer, and was pretrained to replace missing spans marked by sentinel IDs. The official model card says the released checkpoint still needs downstream fine-tuning. The paper reports better robustness to noisy text than mT5.

Sources:

- [ByT5 paper](https://arxiv.org/abs/2105.13626)
- [ByT5-small official model card](https://huggingface.co/google/byt5-small)
- [ByT5-small architecture config](https://huggingface.co/google/byt5-small/blob/main/config.json)
- [T5X model tradeoffs](https://github.com/google-research/t5x/blob/main/docs/models.md#L171-L180)

Test ByT5-small if UMT5 has conditioning failures for unusual Unicode, spelling noise, or emoji. It is smaller and has no unknown-byte path, but raw splicing already gives the exact-inclusion guarantee. Its byte sequences are longer, and official T5X documentation reports inference up to ten times slower depending on the task.

### Qwen3-1.7B

The official card lists 1.7B total parameters, 1.4B non-embedding parameters, 100+ languages and dialects, and Apache-2.0 licensing. It states that non-thinking mode is a hard switch and recommends temperature 0.7, top-p 0.8, and top-k 20 for that mode. Transformers 4.56.2 is newer than the stated minimum of 4.51.0.

Sources:

- [Qwen3-1.7B official model card](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Model size and multilingual support](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/README.md#L28-L45)
- [Non-thinking mode](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/README.md#L145-L161)

The seven current LoRA target names also exist in the Transformers Qwen3 implementation. A Qwen2.5 adapter cannot be used with Qwen3, so an adapter retrain is required. Both training and inference chat templates must set `enable_thinking=False`.

- [Transformers 4.56.2 Qwen3 module names](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py#L70-L82)
- [Transformers 4.56.2 Qwen3 attention](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py#L171-L182)

### Qwen2.5-1.5B-Instruct

This is the lowest-risk base because it already trains and serves here. Its official card lists Chinese among 29+ languages, 1.54B total parameters, 1.31B non-embedding parameters, and Apache-2.0 licensing. It remains suitable if the Qwen3 blind style test does not win.

- [Qwen2.5-1.5B-Instruct official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/README.md#L27-L52)

### Qwen3-4B-Instruct-2507

The official card lists 4.0B parameters and non-thinking-only operation. It reports IFEval 83.4, Creative Writing v3 83.5, and WritingBench 83.4. It is the best quality trial in this set, but not the default for this computer.

- [Qwen3-4B-Instruct-2507 official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/README.md#L34-L72)

At the time of this research, the GPU had only 2,621 MiB free under normal desktop load. The estimated quantized resident weights are about 2.42 GiB before the adapter, KV cache, CUDA workspace, and allocator reserve. Test this model only after other GPU users are closed. Stop the trial if measured peak usage leaves less than 1 GiB headroom.

### Other models

- Gemma 3 supports more than 140 languages, and its 1B model has a 32K context. It is small, but Chinese chat style is not its stated focus and access uses Gemma terms rather than Apache-2.0. Source: [Google Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3).
- InternLM2.5-1.8B-Chat is a credible Chinese model, but its official loading example requires `trust_remote_code=True`. Its card also separates the code license from model-weight commercial terms. Source: [InternLM2.5-1.8B-Chat official card](https://huggingface.co/internlm/internlm2_5-1_8b-chat/blob/main/README.md#L44-L78).
- Qwen3.5-2B has 2B language parameters, a vision encoder, a Gated DeltaNet hybrid architecture, and support for 201 languages and dialects. Its official card requires Transformers from the GitHub `main` branch. This conflicts with the pinned, reproducible Transformers 4.56.2 stack and changes LoRA target assumptions. Sources: [architecture](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/README.md#L40-L69) and [runtime requirement](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/README.md#L684-L691).

## Other model families

### Older T5 infillers

`google/mt5-base` has the same useful encoder-decoder shape and was pretrained on 101 languages including Chinese. It is a good ablation for the updated UMT5 corpus, not the first choice. The official release lists 300 million parameters for mT5-small and 580 million for mT5-base. Source: [mT5 official release](https://github.com/google-research/multilingual-t5#released-model-checkpoints).

### BERT and masked non-autoregressive models

Chinese BERT is only 110 million parameters and uses character tokenization, but its masked-language objective predicts fixed masked positions. It does not natively choose a variable number of output characters on each side of the anchor. Mask-Predict adds target-length prediction and repeated refinement, but its maintained reference is built for machine translation and can depend on an autoregressive teacher. This adds machinery without a strong style-generation prior.

- [BERT Chinese model and masked objective](https://github.com/google-research/bert/blob/master/README.md#L167-L174)
- [Fairseq non-autoregressive models](https://github.com/facebookresearch/fairseq/tree/main/examples/nonautoregressive_translation)

### Insertion Transformer and GLM

An Insertion Transformer can begin with the protected anchor and add tokens anywhere. GLM uses autoregressive blank infilling and has an official 335 million parameter Chinese checkpoint. Both fit the shape of the problem, but their official implementations require different decoders and older, translation-focused or custom training stacks. Neither has a clear advantage over UMT5 for this repository.

- [Insertion Transformer paper](https://proceedings.mlr.press/v97/stern19a.html)
- [GLM paper](https://aclanthology.org/2022.acl-long.26/)
- [GLM official checkpoints](https://github.com/THUDM/GLM#pretrained-models)

### Retrieval, templates, and small language models

Retrieval and fixed templates are the strongest deterministic baseline. Select a safe pattern, insert the raw user text into a known slot, and validate. They need no GPU and preserve style fragments, but they have limited novelty and can copy private source text. A character n-gram, forward and backward recurrent LM, or small Transformer can also generate around a fixed anchor. The corpus is too small to expect strong semantic handling, so use these only as baselines or memorization checks.

- [Retrieve-and-edit generation paper](https://arxiv.org/abs/1709.08878)
- [Character-level recurrent language model study](https://arxiv.org/abs/1506.02078)

### Text diffusion

Current open text diffusion models do not fit this system. LLaDA is 8B, pins Transformers 4.38.2, does not publish its full training framework, and reports slower sampling without a KV cache. Dream is 7B, pins Transformers 4.46.2, and requires at least 20 GB of GPU memory. Both conflict with the 8 GB GPU and pinned stack.

- [LLaDA official repository](https://github.com/ML-GSAI/LLaDA)
- [Dream official requirements](https://github.com/DreamLM/Dream/blob/main/README.md#L29-L34)

## Why the model is not the guarantee

Any generative model predicts likely text rather than proving a string invariant. Prompt compliance and fine-tuning can improve inclusion rate, but they cannot prove 100 percent inclusion. The Rust and Python applications must enforce the contract.

Token equality is also not exact string equality. The local Qwen tokenizer preserved ordinary Chinese, ASCII spaces, newlines, and emoji. It normalized a decomposed accent and removed text that tokenized as `<|endoftext|>` when decoding with `skip_special_tokens=True`. The original string must never be reconstructed by tokenizing and decoding it.

The Transformers tokenizer API documents that `skip_special_tokens` removes special tokens and that `clean_up_tokenization_spaces` can change spaces:

- [Transformers 4.56.2 tokenizer decode API](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3852-L3874)

## Why `force_words_ids` is not the production answer

Transformers 4.56.2 converts `force_words_ids` phrases into `PhrasalConstraint` instances. These enforce ordered token sequences through constrained beam search.

- [GenerationConfig force words definition](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L207-L215)
- [PhrasalConstraint conversion](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L2583-L2621)
- [PhrasalConstraint implementation](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L132-L180)

This path has four problems for this bot:

1. **No sampling:** constrained generation requires more than one beam, and `do_sample=True` is rejected. This removes the current style diversity. Sources: [mode and sampling validation](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/configuration_utils.py#L488-L509) and [beam count requirement](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L480-L489).
2. **The prompt can satisfy the constraint:** constraint state scans the full decoder hypothesis, including the prompt. If the required string is in the prompt, generation can treat the condition as complete before it emits any reply token. Sources: [full-sequence constraint scan](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_constraints.py#L416-L429) and [full hypothesis use](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L697-L721).
3. **Finalization can fail open:** upstream source states that unsuccessful constraints can cause finalization to return the highest-scoring output. Source: [constrained beam finalization fallback](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/beam_search.py#L840-L855).
4. **The path is moving:** Transformers 4.56.2 warns that constrained beam search is moving to a `custom_generate` repository. Source: [versioned warning](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L2580-L2589).

A local diagnostic with the current Qwen2.5 adapter, four beams, and a three-token synthetic Chinese phrase produced an output that contained only the required phrase. Exact inclusion passed, but style quality failed. PyTorch allocated 1,243 MiB and reserved 1,304 MiB. Use constrained beam only as a diagnostic.

## Recommended exact-inclusion design

### Data contract

Send two separate fields from Rust to Python:

```json
{"required_text":"...","seed":42}
```

Rules:

- Keep `required_text` untouched through JSON parsing.
- Reject an empty value only if the product contract forbids it.
- Reject values longer than the downstream Telegram result limit.
- Never silently trim, normalize, or truncate it.
- Treat user text as data, not as a command.

### Dual-sentinel span infilling

Use this path for ByT5-small and UMT5-base:

1. Encode a cleaned conditioning copy between two sentinel tokens.
2. Generate a sequence with the expected three sentinels.
3. Require each sentinel once and in order.
4. Read the first missing span as the left context and the second as the right context.
5. Decode and clean only those generated spans.
6. Build `left + required_text + right` with the untouched Python string.
7. Apply length and format checks, then verify exact inclusion in Python.
8. Return the protected text with the result and verify it again in Rust before Telegram delivery.

The model controls both sides and therefore controls whether the text appears at the start, middle, or end. It does not control the bytes inside the protected span. If sentinel parsing fails, retry with a new seed or return an error. Do not guess span boundaries or return an unverified draft.

### Decoder-only marker generation

Use sampled marker generation so the model can place the input at the start, middle, or end.

1. Choose a marker that does not occur in `required_text`, such as an application-owned nonce.
2. Give the model both the required text as quoted data and an instruction to emit the marker once where that text belongs.
3. Sample normally with Qwen3-1.7B in non-thinking mode.
4. If the draft already contains the exact required text, accept it.
5. Otherwise, require exactly one marker and replace it with the original `required_text` Python object.
6. Apply all formatting and length checks.
7. Verify `required_text in final_reply` in Python.
8. Return both values to Rust and repeat the check immediately before creating the Telegram result.
9. Fail closed if any later transformation removes or changes the required text.

The guarantee comes from direct raw-string splicing and final validation, not from model compliance. A deterministic fallback can place the original text at position zero and sample only a suffix. If prefix placement is not acceptable, return a clear generation error instead of an invalid reply.

### Conditioned training records

Create several anchored records from each source message:

1. Select a non-empty contiguous fragment from the message.
2. Preserve the text before the fragment as `left` and the text after it as `right`.
3. For ByT5 or UMT5, train the dual-sentinel input and target shown above.
4. For Qwen, put the fragment in a structured `required_text` field and replace its target occurrence with one application marker.
5. Preserve the source position. Include start, middle, and end anchors.
6. Split source messages before creating fragment variants, so variants cannot cross the train and validation boundary.
7. Include several fragment lengths and Chinese, ASCII, punctuation, emoji, and whitespace cases.
8. Keep an unconditioned record path only if empty inline queries must still generate a result.

This expansion provides more task examples but not more independent messages. Track validation by original message, and include synthetic user text that never occurs in the corpus to measure out-of-distribution placement.

## Memory plan

Bitsandbytes uses four-bit linear layers, while embeddings and other modules can remain at higher precision. Hugging Face recommends NF4 for four-bit base-model training. Nested quantization saves about 0.4 bits per quantized parameter.

- [Transformers 4.56.2 bitsandbytes guide](https://huggingface.co/docs/transformers/v4.56.2/en/quantization/bitsandbytes)
- [Nested quantization](https://huggingface.co/docs/transformers/v4.56.2/en/quantization/bitsandbytes#nested-quantization)

For tied BF16 embeddings and four-bit non-embedding weights, a lower planning estimate is:

```text
M_weights = 0.5 * P_non_embedding + 2 * (P_total - P_non_embedding) bytes
```

| Model | Formula result | Warm inference plan | QLoRA training plan |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 1.04 GiB | 1.2 to 1.8 GiB | 3.0 to 5.0 GiB |
| Qwen3-1.7B | 1.21 GiB | 1.5 to 2.2 GiB | 3.5 to 5.5 GiB |
| Qwen3-4B-2507 | 2.42 GiB | 3.0 to 4.5 GiB | 5.5 to 7.8 GiB |

The UMT5-base trial used NF4 base weights, rank-16 LoRA, gradient checkpointing, and batch size 1. It reached 2.45 GiB peak allocated VRAM. Frozen BF16 weights remain possible only after a new memory probe. The weight-only lower bounds are about 0.56 GiB for ByT5-small and 1.08 GiB for UMT5-base.

These are estimates with about 1 to 2 GiB uncertainty. CUDA workspaces, allocator reserve, unquantized modules, logits, desktop use, and model-specific kernels can change peak usage. Record `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()` in every model trial.

## Evaluation plan

1. **Contract tests:** use synthetic Chinese, ASCII, leading and trailing spaces, newlines, emoji, zero-width joiners, combining marks, sentinel text, repeated fragments, and values near the length limit. Check exact equality through Python generation, JSON transport, Rust parsing, and final Telegram formatting.
2. **Method comparison:** compare dual-sentinel infilling, Qwen marker splicing, and a character n-gram forward and backward baseline. Every method must reach 100 percent final inclusion by application construction.
3. **Model comparison:** train ByT5-small, UMT5-base, Qwen3-1.7B, and the current Qwen2.5 baseline with the same original-message split, anchor examples, seed set, and update budget where the architectures permit. Use non-thinking mode for every Qwen3 template.
4. **Anchor coverage:** score start, middle, and end placement separately. Include held-out source fragments and synthetic user text that does not occur in any source message.
5. **Blind quality review:** rate held-out results for style fidelity, natural placement, relevance to the required text, repetition, instruction leakage, and privacy leakage. Exact inclusion is a pass gate, not a quality score.
6. **Performance:** measure cold load, warm p50 and p95 latency, generated bytes or tokens per second, peak allocated VRAM, peak reserved VRAM, and free headroom in the persistent worker.
7. **4B gate:** test Qwen3-4B only when the GPU is otherwise free. Promote it only if blind ratings clearly beat the smaller candidates and measured peak use leaves at least 1 GiB headroom.

## Final recommendation

Keep the current Qwen2.5 adapter as the production model. Build exact inclusion once as a model-independent marker-and-splice invariant, then test Qwen3-1.7B as the next naturalness challenger. The completed UMT5-base trial reached 100 percent structured inclusion but produced poor Chinese joins, so do not deploy it. Test ByT5-small only for a focused noisy-input ablation. Change the production model only after a blind held-out comparison.

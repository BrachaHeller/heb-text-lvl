"""O(1) 3-tier cascading dictionary lookups and cross-lingual Zipf rarity extraction."""

import itertools
import logging
import math
import re
import string
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, MarianMTModel, MarianTokenizer

from hebrew_readability.hf_hub import download_frequency_dictionary

logger = logging.getLogger(__name__)

_NLP_EN: Optional[Any] = None
_XLM_TOKENIZER: Optional[Any] = None
_XLM_MODEL: Optional[Any] = None
_MARIAN_TOKENIZER: Optional[Any] = None
_MARIAN_MODEL: Optional[Any] = None
_CACHED_DICTS: Optional[Tuple[Tuple[Dict, ...], int]] = None


def _get_alignment_models():
    """Lazy singleton loader for SpaCy, MarianMT, and XLM-RoBERTa."""
    global _NLP_EN, _XLM_TOKENIZER, _XLM_MODEL, _MARIAN_TOKENIZER, _MARIAN_MODEL
    if _NLP_EN is None:
        try:
            _NLP_EN = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _NLP_EN = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        xlm_name = "xlm-roberta-base"
        _XLM_TOKENIZER = AutoTokenizer.from_pretrained(xlm_name)
        _XLM_MODEL = (
            AutoModel.from_pretrained(
                xlm_name,
                output_attentions=True,
                output_hidden_states=True,
                torch_dtype=torch.float16,
            )
            .to("cpu")
            .eval()
        )

        marian_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
        _MARIAN_TOKENIZER = MarianTokenizer.from_pretrained(marian_name)
        _MARIAN_MODEL = (
            MarianMTModel.from_pretrained(marian_name, torch_dtype=torch.float16)
            .to("cpu")
            .eval()
        )
    return _NLP_EN, _XLM_TOKENIZER, _XLM_MODEL, _MARIAN_TOKENIZER, _MARIAN_MODEL


def load_cascading_dictionaries(
    csv_path: Optional[str] = None,
) -> Tuple[Tuple[Dict, ...], int]:
    """Loads O(1) 3-tier hash maps (Tuple -> Morphosyntactic POS -> Lemma) from HF Hub or local file."""
    global _CACHED_DICTS
    if _CACHED_DICTS is not None and csv_path is None:
        return _CACHED_DICTS

    resolved_path = download_frequency_dictionary(local_override=csv_path)
    if str(resolved_path).endswith(".parquet"):
        df = pd.read_parquet(resolved_path, engine="pyarrow").fillna("")
    else:
        df = pd.read_csv(resolved_path, keep_default_na=False)
        
    for col in ["rank_tuple", "rank_lemma_pos", "rank_lemma"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(999999).astype(int)
    for col in ["entry_corpus_ratio_%", "pos_corpus_ratio_%", "lemma_corpus_ratio_%"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    # 1. Tuple Domain
    df_tuple = df.sort_values("rank_tuple")
    dict_full = df_tuple.set_index(["lemma_heb", "upos_heb", "lemma_eng"])[
        ["rank_tuple", "entry_corpus_ratio_%"]
    ].to_dict("index")
    dict_fallback_tuple_pos = (
        df_tuple.drop_duplicates(subset=["lemma_heb", "upos_heb"])
        .set_index(["lemma_heb", "upos_heb"])[["rank_tuple", "entry_corpus_ratio_%"]]
        .to_dict("index")
    )
    dict_fallback_tuple_lemma = (
        df_tuple.drop_duplicates(subset=["lemma_heb"])
        .set_index("lemma_heb")[["rank_tuple", "entry_corpus_ratio_%"]]
        .to_dict("index")
    )

    # 2. Morphosyntactic POS Domain
    df_pos = df.sort_values("rank_lemma_pos")
    dict_pos_exact = (
        df_pos.drop_duplicates(subset=["lemma_heb", "upos_heb"])
        .set_index(["lemma_heb", "upos_heb"])[["rank_lemma_pos", "pos_corpus_ratio_%"]]
        .to_dict("index")
    )
    dict_fallback_pos_lemma = (
        df_pos.drop_duplicates(subset=["lemma_heb"])
        .set_index("lemma_heb")[["rank_lemma_pos", "pos_corpus_ratio_%"]]
        .to_dict("index")
    )

    # 3. Lemma Domain
    df_lemma = df.sort_values("rank_lemma")
    dict_lemma_exact = (
        df_lemma.drop_duplicates(subset=["lemma_heb"])
        .set_index("lemma_heb")[["rank_lemma", "lemma_corpus_ratio_%"]]
        .to_dict("index")
    )

    max_rank = int(
        max(df["rank_tuple"].max(), df["rank_lemma_pos"].max(), df["rank_lemma"].max())
    )
    cascading_dicts = (
        dict_full,
        dict_fallback_tuple_pos,
        dict_fallback_tuple_lemma,
        dict_pos_exact,
        dict_fallback_pos_lemma,
        dict_lemma_exact,
    )
    _CACHED_DICTS = (cascading_dicts, max_rank)
    return cascading_dicts, max_rank


def sinkhorn_knopp(matrix: torch.Tensor, max_iters: int = 20) -> torch.Tensor:
    """Iterative doubly-stochastic matrix normalization for word alignment."""
    P = matrix.clone() + 1e-9
    for _ in range(max_iters):
        P = P / P.sum(dim=1, keepdim=True)
        P = P / P.sum(dim=0, keepdim=True)
    return P


def is_valid_pos_match(heb_pos: str, eng_pos: str, heb_lemma: str) -> bool:
    """Checks cross-lingual POS compatibility between Hebrew and English lemmas."""
    if str(heb_lemma) in {"ב", "ה", "ו", "כ", "ל", "מ", "ש"}:
        return True
    if heb_pos == "VERB":
        return eng_pos in {"VERB", "AUX", "NOUN", "ADJ", "ADV"}
    if heb_pos == "AUX":
        return eng_pos in {"AUX", "VERB"}
    if heb_pos in {"NOUN", "PROPN"}:
        return eng_pos in {"NOUN", "PROPN", "PRON", "VERB", "ADJ", "ADV"}
    if heb_pos == "ADJ":
        return eng_pos in {"ADJ", "VERB", "NOUN", "ADV"}
    if heb_pos == "ADV":
        return eng_pos in {"ADV", "PART", "NOUN", "VERB", "ADJ", "PRON", "DET"}
    return eng_pos in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


def chunk_parsed_tokens(
    tokens: List[Dict[str, Any]], max_words: int = 60
) -> Iterator[List[Dict[str, Any]]]:
    """Slices DictaBERT tokens at natural punctuation boundaries to respect VRAM limits."""
    while tokens:
        if len(tokens) <= max_words:
            yield tokens
            break

        p1_idx, p2_idx, p3_idx = -1, -1, -1
        for i in range(max_words - 1, -1, -1):
            word = tokens[i].get("token", "")
            if word in [".", "!", "?"] and p1_idx == -1:
                p1_idx = i
            if word == ";" and p2_idx == -1:
                p2_idx = i
            if word == "," and p3_idx == -1:
                p3_idx = i

        cut_idx = (
            p1_idx
            if p1_idx != -1
            else (p2_idx if p2_idx != -1 else (p3_idx if p3_idx != -1 else max_words - 1))
        )
        yield tokens[: cut_idx + 1]
        tokens = tokens[cut_idx + 1 :]


def extract_lexical_frequency_features(
    clean_text: str,
    dictabert_raw_tokens: List[Dict[str, Any]],
    cascading_dicts: Optional[Tuple[Dict, ...]] = None,
    max_rank: Optional[int] = None,
    oov_threshold: int = 3000,
) -> Tuple[float, float, float, float, float, float, float, str]:
    """Extracts 3-tier OOV and Zipf rarity features using O(1) hash map lookups."""
    if not clean_text or not dictabert_raw_tokens:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "[]"

    if cascading_dicts is None or max_rank is None:
        cascading_dicts, max_rank = load_cascading_dictionaries()

    nlp_en, xlm_tokenizer, xlm_model, marian_tokenizer, marian_model = _get_alignment_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        dict_full,
        dict_fallback_tuple_pos,
        dict_fallback_tuple_lemma,
        dict_pos_exact,
        dict_fallback_pos_lemma,
        dict_lemma_exact,
    ) = cascading_dicts

    paragraph_tuples = []
    oov_tuple_count, zipf_rarity_sum_tuple = 0, 0.0
    oov_pos_count, zipf_rarity_sum_pos = 0, 0.0
    oov_lemma_count, zipf_rarity_sum_lemma = 0, 0.0
    content_words_count = 0

    content_pos_set = {"NOUN", "VERB", "ADJ", "ADV"}
    prefix_lemmas = {"ב", "ה", "ו", "כ", "ל", "מ", "ש"}

    def get_zipf_rarity(ratio_percent: float) -> float:
        return 7.0 if ratio_percent <= 0.0 else max(0.0, -math.log10(ratio_percent))

    for chunk in chunk_parsed_tokens(dictabert_raw_tokens, max_words=60):
        alignable_heb = []
        for token in chunk:
            word_form = token.get("token", "").strip()
            upos = token.get("morph", {}).get("pos", "").strip()
            lemma = token.get("lex", "").strip()

            if (
                not word_form
                or upos in ["PUNCT", "SYM"]
                or all(c in string.punctuation for c in word_form)
            ):
                continue
            if lemma in ["_", "[BLANK]", "<unk>", "None", ""] or all(
                c in string.punctuation for c in lemma
            ):
                lemma = word_form

            alignable_heb.append({"word_form": word_form, "lemma": lemma, "upos": upos})

        if not alignable_heb:
            continue

        chunk_text = " ".join([r["word_form"] for r in alignable_heb])

        marian_model.to(device)
        inputs = marian_tokenizer(
            [chunk_text], return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            translated_tokens = marian_model.generate(**inputs, max_length=256)
        eng_text = marian_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        marian_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eng_doc = nlp_en(eng_text)
        eng_rows = [
            {"text": t.text, "lemma": t.lemma_, "pos": t.pos_}
            for t in eng_doc
            if t.pos_ != "PUNCT"
        ]

        if not eng_rows:
            for h_row in alignable_heb:
                clean_lemma_preview = re.sub(r"[^\u05D0-\u05EA]", "", h_row["lemma"])
                if (
                    clean_lemma_preview
                    and h_row["upos"] in content_pos_set
                    and clean_lemma_preview not in prefix_lemmas
                    and clean_lemma_preview != "של"
                ):
                    l_heb, u_heb = clean_lemma_preview, h_row["upos"]
                    content_words_count += 1

                    full_hit = dict_fallback_tuple_pos.get(
                        (l_heb, u_heb)
                    ) or dict_fallback_tuple_lemma.get(l_heb)
                    pos_hit = dict_pos_exact.get((l_heb, u_heb)) or dict_fallback_pos_lemma.get(
                        l_heb
                    )
                    lemma_hit = dict_lemma_exact.get(l_heb)

                    if (full_hit["rank_tuple"] if full_hit else max_rank + 1) > oov_threshold:
                        oov_tuple_count += 1
                    if (pos_hit["rank_lemma_pos"] if pos_hit else max_rank + 1) > oov_threshold:
                        oov_pos_count += 1
                    if (lemma_hit["rank_lemma"] if lemma_hit else max_rank + 1) > oov_threshold:
                        oov_lemma_count += 1

                    zipf_rarity_sum_tuple += get_zipf_rarity(
                        full_hit["entry_corpus_ratio_%"] if full_hit else 0.0
                    )
                    zipf_rarity_sum_pos += get_zipf_rarity(
                        pos_hit["pos_corpus_ratio_%"] if pos_hit else 0.0
                    )
                    zipf_rarity_sum_lemma += get_zipf_rarity(
                        lemma_hit["lemma_corpus_ratio_%"] if lemma_hit else 0.0
                    )
            continue

        xlm_model.to(device)
        heb_words = [r["word_form"] for r in alignable_heb]
        eng_words = [r["text"] for r in eng_rows]

        tok_src = [xlm_tokenizer.tokenize(w) for w in heb_words]
        tok_tgt = [xlm_tokenizer.tokenize(w) for w in eng_words]

        flat_src = list(
            itertools.chain(*[xlm_tokenizer.convert_tokens_to_ids(x) for x in tok_src])
        )
        flat_tgt = list(
            itertools.chain(*[xlm_tokenizer.convert_tokens_to_ids(x) for x in tok_tgt])
        )

        bpe2word_src = [idx for idx, wl in enumerate(tok_src) for _ in wl][: len(flat_src)]
        bpe2word_tgt = [idx for idx, wl in enumerate(tok_tgt) for _ in wl][: len(flat_tgt)]

        input_ids = (
            [xlm_tokenizer.cls_token_id]
            + flat_src
            + [xlm_tokenizer.sep_token_id]
            + flat_tgt
            + [xlm_tokenizer.sep_token_id]
        )
        input_tensor = torch.tensor([input_ids]).to(device)

        amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
            outputs = xlm_model(input_tensor)
            layer_attns = torch.stack([outputs.attentions[l] for l in [7, 8, 9]]).mean(dim=0)
            pooled_hiddens = torch.stack(
                [outputs.hidden_states[l] for l in [7, 8, 9, 10]]
            ).mean(dim=0)

        heb_end = 1 + len(flat_src)
        eng_start = heb_end + 1
        eng_end = eng_start + len(flat_tgt)

        attn_matrix = layer_attns[0, :, 1:heb_end, eng_start:eng_end].mean(dim=0)
        h_heb = F.normalize(pooled_hiddens[0, 1:heb_end], p=2, dim=1)
        h_eng = F.normalize(pooled_hiddens[0, eng_start:eng_end], p=2, dim=1)

        cosine_matrix = torch.matmul(h_heb, h_eng.t())
        attn_norm = (attn_matrix - attn_matrix.min()) / (
            attn_matrix.max() - attn_matrix.min() + 1e-9
        )
        fused_matrix = (0.6 * attn_norm) + (0.4 * cosine_matrix)
        prob_matrix = sinkhorn_knopp(fused_matrix)

        word_matrix = torch.zeros((len(heb_words), len(eng_words))).to(device)
        counts = torch.zeros_like(word_matrix)
        for i, w_src in enumerate(bpe2word_src):
            for j, w_tgt in enumerate(bpe2word_tgt):
                word_matrix[w_src, w_tgt] += prob_matrix[i, j].item()
                counts[w_src, w_tgt] += 1
        word_matrix = torch.div(word_matrix, counts + 1e-9)

        xlm_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for h_idx, h_row in enumerate(alignable_heb):
            best_eng_idx = int(torch.argmax(word_matrix[h_idx, :]).item())
            eng_pos = eng_rows[best_eng_idx]["pos"]
            raw_eng_lemma = eng_rows[best_eng_idx]["lemma"]

            clean_eng_lemma = re.sub(r"[^a-z]", "", raw_eng_lemma.lower())
            clean_heb_lemma = re.sub(r"[^\u05D0-\u05EA]", "", h_row["lemma"])

            if not clean_heb_lemma:
                continue

            final_eng = (
                clean_eng_lemma
                if is_valid_pos_match(h_row["upos"], eng_pos, clean_heb_lemma)
                else "unaligned"
            )
            paragraph_tuples.append((clean_heb_lemma, h_row["upos"], final_eng))

            if (
                h_row["upos"] not in content_pos_set
                or clean_heb_lemma in prefix_lemmas
                or clean_heb_lemma == "של"
            ):
                continue

            content_words_count += 1
            l_heb, u_heb = clean_heb_lemma, h_row["upos"]

            # O(1) 3-tier lookup
            if (l_heb, u_heb, final_eng) in dict_full:
                full_hit = dict_full[(l_heb, u_heb, final_eng)]
            elif (l_heb, u_heb) in dict_fallback_tuple_pos:
                full_hit = dict_fallback_tuple_pos[(l_heb, u_heb)]
            else:
                full_hit = dict_fallback_tuple_lemma.get(l_heb)

            if (l_heb, u_heb) in dict_pos_exact:
                pos_hit = dict_pos_exact[(l_heb, u_heb)]
            else:
                pos_hit = dict_fallback_pos_lemma.get(l_heb)

            lemma_hit = dict_lemma_exact.get(l_heb)

            tuple_rank = full_hit["rank_tuple"] if full_hit else max_rank + 1
            pos_rank = pos_hit["rank_lemma_pos"] if pos_hit else max_rank + 1
            lemma_rank = lemma_hit["rank_lemma"] if lemma_hit else max_rank + 1

            if tuple_rank > oov_threshold:
                oov_tuple_count += 1
            if pos_rank > oov_threshold:
                oov_pos_count += 1
            if lemma_rank > oov_threshold:
                oov_lemma_count += 1

            zipf_rarity_sum_tuple += get_zipf_rarity(
                full_hit["entry_corpus_ratio_%"] if full_hit else 0.0
            )
            zipf_rarity_sum_pos += get_zipf_rarity(
                pos_hit["pos_corpus_ratio_%"] if pos_hit else 0.0
            )
            zipf_rarity_sum_lemma += get_zipf_rarity(
                lemma_hit["lemma_corpus_ratio_%"] if lemma_hit else 0.0
            )

    oov_r_tuple = oov_tuple_count / content_words_count if content_words_count > 0 else 0.0
    oov_r_pos = oov_pos_count / content_words_count if content_words_count > 0 else 0.0
    oov_r_lemma = oov_lemma_count / content_words_count if content_words_count > 0 else 0.0

    w_rarity_tuple = (
        zipf_rarity_sum_tuple / content_words_count if content_words_count > 0 else 0.0
    )
    w_rarity_pos = zipf_rarity_sum_pos / content_words_count if content_words_count > 0 else 0.0
    w_rarity_lemma = (
        zipf_rarity_sum_lemma / content_words_count if content_words_count > 0 else 0.0
    )
    w_rarity_tuple_norm = w_rarity_tuple / 7.0 if w_rarity_tuple > 0 else 0.0

    return (
        round(oov_r_tuple, 4),
        round(oov_r_pos, 4),
        round(oov_r_lemma, 4),
        round(w_rarity_tuple, 4),
        round(w_rarity_pos, 4),
        round(w_rarity_lemma, 4),
        round(w_rarity_tuple_norm, 4),
        str(paragraph_tuples),
    )
"""Syntactic dependency tree and morphological POS extractors via Stanza and DictaBERT."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import stanza
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

_STANZA_PIPELINE: Optional[stanza.Pipeline] = None
_DICTABERT_TOKENIZER: Optional[ Any ] = None
_DICTABERT_MODEL: Optional[ Any ] = None


def _get_stanza_pipeline() -> stanza.Pipeline:
    """Lazy singleton loader for Hebrew Stanza pipeline."""
    global _STANZA_PIPELINE
    if _STANZA_PIPELINE is None:
        stanza.download("he", verbose=False)
        _STANZA_PIPELINE = stanza.Pipeline(
            "he",
            processors="tokenize,mwt,pos,lemma,ner,depparse",
            use_gpu=torch.cuda.is_available(),
            verbose=False,
        )
    return _STANZA_PIPELINE


def _get_dictabert() -> Tuple[Any, Any]:
    """Lazy singleton loader for DictaBERT large parse model."""
    global _DICTABERT_TOKENIZER, _DICTABERT_MODEL
    if _DICTABERT_MODEL is None:
        model_name = "dicta-il/dictabert-large-parse"
        _DICTABERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _DICTABERT_MODEL = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _DICTABERT_MODEL.eval()
        if torch.cuda.is_available():
            _DICTABERT_MODEL = _DICTABERT_MODEL.to("cuda")
    return _DICTABERT_TOKENIZER, _DICTABERT_MODEL


def get_stanza_features(
    clean_text: str,
) -> Tuple[int, float, float, float, float, float]:
    """Extracts sentence count, physical/linguistic ASL, clause density, tree depth, and passive verb ratio."""
    if not clean_text:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0

    nlp_stanza = _get_stanza_pipeline()
    doc = nlp_stanza(clean_text)
    total_sentences = len(doc.sentences)
    if total_sentences == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0

    total_phys_words, total_ling_words, total_clauses, sum_tree_depth = 0, 0, 0, 0
    verb_aux_count, passive_count = 0, 0

    for sent in doc.sentences:
        sent_max_depth = 0
        head_map = {word.id: word.head for word in sent.words}
        sent_clause_heads = []

        for token in sent.tokens:
            if not all(w.upos in ["PUNCT", "SYM"] for w in token.words):
                total_phys_words += 1

        for word in sent.words:
            upos = word.upos or ""
            deprel = word.deprel or ""
            feats = word.feats or ""

            if upos in ["PUNCT", "SYM"]:
                continue

            total_ling_words += 1

            if upos in ["VERB", "AUX"]:
                verb_aux_count += 1
                if any(
                    m in feats
                    for m in [
                        "Voice=Pass",
                        "HebBinyan=PUAL",
                        "HebBinyan=HUFAL",
                        "HebBinyan=NIFAL",
                    ]
                ):
                    passive_count += 1

            if deprel != "xcomp" and "VerbForm=Inf" not in feats:
                if deprel in ["root", "ccomp", "advcl"]:
                    sent_clause_heads.append(word.id)
                elif deprel in ["acl", "acl:relcl", "conj"] and upos in ["VERB", "AUX"]:
                    sent_clause_heads.append(word.id)

            current_depth, current_id, visited = 0, word.id, set()
            while current_id != 0 and current_id not in visited:
                visited.add(current_id)
                current_id = head_map.get(current_id, 0)
                current_depth += 1
            sent_max_depth = max(sent_max_depth, current_depth)

        total_clauses += max(1, len(sent_clause_heads))
        sum_tree_depth += sent_max_depth

    asl_phys = round(total_phys_words / total_sentences, 2)
    asl_ling = round(total_ling_words / total_sentences, 2)
    avg_clauses = round(total_clauses / total_sentences, 2)
    avg_tree_depth = round(sum_tree_depth / total_sentences, 2)
    ratio_passive = round(passive_count / verb_aux_count, 4) if verb_aux_count > 0 else 0.0

    return total_sentences, asl_phys, asl_ling, avg_clauses, avg_tree_depth, ratio_passive


def get_dictabert_features(
    clean_text: str,
) -> Tuple[float, float, float, List[Dict[str, Any]]]:
    """Extracts adjective, adverb, and pronoun ratios alongside raw DictaBERT tokens."""
    if not clean_text:
        return 0.0, 0.0, 0.0, []

    dct_tokenizer, dct_model = _get_dictabert()
    try:
        parsed_doc = dct_model.predict([clean_text], dct_tokenizer)
        if not parsed_doc or "tokens" not in parsed_doc[0]:
            return 0.0, 0.0, 0.0, []

        tokens = parsed_doc[0]["tokens"]
        ling_pos = {"ADJ": 0, "ADV": 0, "PRON": 0, "NOMINAL": 0}
        total_ling_words = 0

        for token in tokens:
            core_pos = token["morph"]["pos"]
            if core_pos in ["PUNCT", "SYM"]:
                continue

            total_ling_words += 1
            if core_pos == "ADJ":
                ling_pos["ADJ"] += 1
            if core_pos == "ADV":
                ling_pos["ADV"] += 1
            if core_pos == "PRON":
                ling_pos["PRON"] += 1
            if core_pos in ["NOUN", "PROPN", "PRON"]:
                ling_pos["NOMINAL"] += 1

            total_ling_words += len(token["morph"].get("prefixes", []))
            suffix = token["morph"].get("suffix")
            if suffix:
                total_ling_words += 1
                if suffix == "PRON":
                    ling_pos["PRON"] += 1
                    ling_pos["NOMINAL"] += 1
                elif suffix in ["NOUN", "PROPN"]:
                    ling_pos["NOMINAL"] += 1

        ratio_adj = round(ling_pos["ADJ"] / total_ling_words, 4) if total_ling_words > 0 else 0.0
        ratio_adv = round(ling_pos["ADV"] / total_ling_words, 4) if total_ling_words > 0 else 0.0
        ratio_pron = (
            round(ling_pos["PRON"] / ling_pos["NOMINAL"], 4) if ling_pos["NOMINAL"] > 0 else 0.0
        )

        return ratio_adj, ratio_adv, ratio_pron, tokens

    except Exception as exc:
        logger.warning(f"DictaBERT Error: {exc}")
        return 0.0, 0.0, 0.0, []
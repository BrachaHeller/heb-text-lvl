"""Unified single-text and checkpointed batch feature extraction orchestrator."""

import logging
import os
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from features.lexical_rarity import (
    extract_lexical_frequency_features,
    load_cascading_dictionaries,
)
from features.syntactic_morph import (
    get_dictabert_features,
    get_stanza_features,
)
from features.typographical_volume import (
    get_base_features,
    get_syllable_features,
)

from hebrew_readability.config import read_table, save_table

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Orchestrates Hebrew linguistic feature extraction for single strings and batch DataFrames."""

    def __init__(self, dictionary_csv_path: Optional[str] = None, oov_threshold: int = 3000):
        self.cascading_dicts, self.max_rank = load_cascading_dictionaries(dictionary_csv_path)
        self.oov_threshold = oov_threshold

    def extract_single_text(self, raw_text: str) -> pd.DataFrame:
        """Extracts raw linguistic metrics for a single Hebrew string into a 1-row DataFrame."""
        nikkud_ratio, clean_text, num_phys_words = get_base_features(raw_text)
        (
            stz_sents,
            stz_asl_phys,
            stz_asl_ling,
            stz_avg_clauses,
            stz_avg_depth,
            stz_ratio_pass,
        ) = get_stanza_features(clean_text)
        dct_ratio_adj, dct_ratio_adv, dct_ratio_pron, dictabert_tokens = get_dictabert_features(
            clean_text
        )
        (
            oov_r_tuple,
            oov_r_pos,
            oov_r_lemma,
            w_rarity_tuple,
            w_rarity_pos,
            w_rarity_lemma,
            w_rarity_tuple_norm,
            lexical_tuples,
        ) = extract_lexical_frequency_features(
            clean_text=clean_text,
            dictabert_raw_tokens=dictabert_tokens,
            cascading_dicts=self.cascading_dicts,
            max_rank=self.max_rank,
            oov_threshold=self.oov_threshold,
        )
        vocalized_text, total_syl, avg_syl_word, avg_syl_sent = get_syllable_features(
            clean_text, num_phys_words, stz_sents
        )

        row_dict = {
            "1_nikkud_ratio": nikkud_ratio,
            "2_cleaned_text": clean_text,
            "3_num_physical_words": num_phys_words,
            "4_stz_num_sentences": stz_sents,
            "5_stz_avg_phys_words_per_sent": stz_asl_phys,
            "6_stz_avg_lemmas_per_sent": stz_asl_ling,
            "7_stz_avg_clauses_per_sent": stz_avg_clauses,
            "8_stz_avg_tree_depth_per_sent": stz_avg_depth,
            "9_stz_ratio_passive_verbs": stz_ratio_pass,
            "10_dct_ratio_adj": dct_ratio_adj,
            "11_dct_ratio_adv": dct_ratio_adv,
            "12_dct_ratio_pron": dct_ratio_pron,
            "13a_oov_ratio_tuple": oov_r_tuple,
            "13b_oov_ratio_pos": oov_r_pos,
            "13c_oov_ratio_lemma": oov_r_lemma,
            "14_sipf_rarity_tuple_norm": w_rarity_tuple_norm,
            "14a_zipf_rarity_tuple": w_rarity_tuple,
            "14b_zipf_rarity_pos": w_rarity_pos,
            "14c_zipf_rarity_lemma": w_rarity_lemma,
            "15_vocalized_text": vocalized_text,
            "16_total_syllables": total_syl,
            "17_avg_syllables_per_word": avg_syl_word,
            "18_avg_syllables_per_sentence": avg_syl_sent,
            "19_debug_lexical_tuples": lexical_tuples,
        }
        return pd.DataFrame([row_dict])

    def extract_batch_df(
        self,
        df: pd.DataFrame,
        text_column: str = "raw_text",
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 50,
    ) -> pd.DataFrame:
        """Runs batch feature extraction with automatic disk checkpointing and VRAM cleanup."""
        start_idx = 0
        output_rows = []

        if checkpoint_path and os.path.exists(checkpoint_path):
            existing_df = read_table(checkpoint_path)
            output_rows = existing_df.to_dict("records")
            start_idx = len(output_rows)
            logger.info(f"Resuming batch extraction from row {start_idx}/{len(df)}...")

        for idx in tqdm(range(start_idx, len(df)), desc="Extracting Hebrew Features"):
            row = df.iloc[idx]
            raw_text = str(row[text_column]) if pd.notnull(row[text_column]) else ""
            feat_df = self.extract_single_text(raw_text)

            enriched_row = row.to_dict()
            enriched_row.update(feat_df.iloc[0].to_dict())
            output_rows.append(enriched_row)

            if (idx + 1) % checkpoint_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if checkpoint_path:
                    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                    save_table(pd.DataFrame(output_rows), checkpoint_path)
                    logger.info(f"Saved checkpoint at row {idx + 1} -> {checkpoint_path}")

        final_df = pd.DataFrame(output_rows)
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            save_table(final_df, checkpoint_path)
        return final_df
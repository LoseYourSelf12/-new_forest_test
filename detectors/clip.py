from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
import torch
import re
import open_clip
from open_clip import get_tokenizer
from PIL import Image
from .base import Detector


def init_clip(model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = get_tokenizer(model_name)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, preprocess, tokenizer


def enc77_batch(texts: List[str], tokenizer, device: Optional[str] = None) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids: List[List[int]] = []
    for t in texts:
        token_ids = tokenizer.encode(t.lower())[:77]
        token_ids += [0] * (77 - len(token_ids))
        ids.append(token_ids)
    if not ids:
        return torch.empty((0, 77), dtype=torch.long, device=device)
    return torch.tensor(ids, device=device)


def get_clip_features(
    text: str = '',
    model: Optional[torch.nn.Module] = None,
    preprocess: Any = None,
    tokenizer: Any = None,
    pos_prompts: Optional[List[str]] = None,
    neg_prompts: Optional[List[str]] = None,
    core_phrases: Optional[List[str]] = None,
    aux_phrases: Optional[List[str]] = None,
    include_text_flags: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:

    text = text or ''
    low = text.lower()

    if device is None:
        if model is not None:
            device = next(model.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features: List[np.ndarray] = []

    if model is not None and tokenizer is not None:
        with torch.no_grad():
            txt_toks = enc77_batch([text], tokenizer, device)
            emb = model.encode_text(txt_toks)[0]
            emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
        features.append(emb)

        if pos_prompts:
            with torch.no_grad():
                pos_toks = enc77_batch(pos_prompts, tokenizer, device)
                pos_emb = model.encode_text(pos_toks)
                pos_emb = torch.nn.functional.normalize(pos_emb, dim=-1)
            pos_sim = emb @ pos_emb.cpu().numpy().T
            features.append(pos_sim)

        if neg_prompts:
            with torch.no_grad():
                neg_toks = enc77_batch(neg_prompts, tokenizer, device)
                neg_emb = model.encode_text(neg_toks)
                neg_emb = torch.nn.functional.normalize(neg_emb, dim=-1)
            neg_sim = emb @ neg_emb.cpu().numpy().T
            features.append(neg_sim)

    if core_phrases:
        core_flags = np.array([1 if phrase in low else 0 for phrase in core_phrases], dtype=int)
        features.append(core_flags)
    if aux_phrases:
        aux_flags = np.array([1 if phrase in low else 0 for phrase in aux_phrases], dtype=int)
        features.append(aux_flags)

    if include_text_flags:
        has_email = int(bool(re.search(r"[\w\.-]+@[\w\.-]+", low)))
        ru_domain = int(bool(re.search(r"\.ru\b", low)))
        has_phone = int(bool(re.search(r"\+?\d[\d\s\-]{8,}\d", low)))
        flags = np.array([has_email, ru_domain, has_phone], dtype=int)
        features.append(flags)

    if not features:
        return np.array([])

    return np.concatenate(features, axis=0)


class ClipDetector(Detector):
    """Detector based on CLIP embeddings."""

    def __init__(
        self,
        name: str,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        pos_prompts: Optional[List[str]] = None,
        neg_prompts: Optional[List[str]] = None,
        core_phrases: Optional[List[str]] = None,
        aux_phrases: Optional[List[str]] = None,
        include_text_flags: bool = False,
    ):
        self._model, self._preprocess, self._tokenizer = init_clip(model_name, pretrained)
        self._pos_prompts = pos_prompts
        self._neg_prompts = neg_prompts
        self._core_phrases = core_phrases
        self._aux_phrases = aux_phrases
        self._include_text_flags = include_text_flags
        feature_len = 512
        if pos_prompts:
            feature_len += len(pos_prompts)
        if neg_prompts:
            feature_len += len(neg_prompts)
        if core_phrases:
            feature_len += len(core_phrases)
        if aux_phrases:
            feature_len += len(aux_phrases)
        if include_text_flags:
            feature_len += 3
        super().__init__(name=name, features=[f"f{i}" for i in range(feature_len)])

    def __call__(self, local: Dict[str, Any]) -> None:
        text = local.get('clip_text') or local.get('qwen_text') or local.get('ocr_text') or local.get('txt', '')
        vec = get_clip_features(
            text=text,
            model=self._model,
            preprocess=self._preprocess,
            tokenizer=self._tokenizer,
            pos_prompts=self._pos_prompts,
            neg_prompts=self._neg_prompts,
            core_phrases=self._core_phrases,
            aux_phrases=self._aux_phrases,
            include_text_flags=self._include_text_flags,
        )
        self._vec = vec

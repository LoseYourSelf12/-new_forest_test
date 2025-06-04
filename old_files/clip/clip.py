import pandas as pd
import numpy as np
import open_clip
from open_clip import get_tokenizer
import torch
from typing import List, Optional, Tuple, Any, Dict
import re


def load_qwen_txt(
    local: Dict[str, Any],
    qwen_file: str = '/home/tbdbj/forest_test/qwen/qwen_ds_3.csv',
    delimiter: str = ';'
) -> str:

    file_name = local.get('current_file_name', local.get('file_name'))
    try:
        df = pd.read_csv(qwen_file, encoding='utf-8', delimiter=delimiter)
        match = df.loc[df['file_name'] == file_name]
        if not match.empty:
            return match.iloc[0]['texts']
    except Exception as e:
        print(f"Error: {e}")
    return ''


def init_clip(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: Optional[str] = None
) -> Tuple[torch.nn.Module, Any, Any]:
    
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = get_tokenizer(model_name)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, preprocess, tokenizer


def enc77_batch(
    texts: List[str],
    tokenizer: Any,
    device: Optional[str] = None
) -> torch.Tensor:
    
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
    device: Optional[str] = None
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

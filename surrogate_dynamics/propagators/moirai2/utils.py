from typing import Optional

import numpy as np
import torch
from jaxtyping import Bool, Float, Int


def packed_attention_mask(
    sample_id: Int[torch.Tensor, "*batch seq_len"], # pyright: ignore
) -> Bool[torch.Tensor, "*batch seq_len seq_len"]: # pyright: ignore
    sample_id = sample_id.unsqueeze(-1)
    attention_mask = sample_id.eq(sample_id.mT)
    return attention_mask

def packed_causal_attention_mask(
    sample_id: Int[torch.Tensor, "*batch seq_len"], # pyright: ignore
    time_id: Int[torch.Tensor, "*batch seq_len"], # pyright: ignore
) -> Bool[torch.Tensor, "*batch seq_len seq_len"]: # pyright: ignore
    attention_mask = packed_attention_mask(sample_id)
    expanded_id1 = time_id.unsqueeze(-2)
    expanded_id2 = time_id.unsqueeze(-1)
    compare_res = expanded_id1 <= expanded_id2
    attention_mask = attention_mask * compare_res
    return attention_mask

def safe_div(
    numer: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    return numer / torch.where(
        denom == 0,
        1.0,
        denom,
    )

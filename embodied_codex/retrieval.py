"""Small deterministic retrieval layer shared by all capability assets."""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping


def _tokens(value: Any):
    text=str(value or "").casefold()
    words=re.findall(r"[a-z0-9_]+",text)
    cjk=re.findall(r"[\u3400-\u9fff]",text)
    return words+cjk+["".join(cjk[index:index+2]) for index in range(max(0,len(cjk)-1))]


def rank_records(query: str, records: Iterable[Mapping[str,Any]], *,
                 text_fields: tuple[str,...], id_field: str, limit: int=8):
    rows=[dict(item) for item in records];query_tokens=set(_tokens(query))
    documents=[];frequencies={}
    for row in rows:
        tokens=[]
        for field in text_fields:tokens.extend(_tokens(row.get(field)))
        token_set=set(tokens);documents.append((row,tokens,token_set))
        for token in token_set:frequencies[token]=frequencies.get(token,0)+1
    ranked=[];count=max(1,len(rows));phrase=str(query).casefold().strip()
    for row,tokens,token_set in documents:
        overlap=query_tokens.intersection(token_set)
        score=sum(math.log((count+1)/(frequencies[token]+0.5))+1 for token in overlap)
        body=" ".join(str(row.get(field) or "").casefold() for field in text_fields)
        if phrase and phrase in body:score+=4
        score+=0.05*sum(tokens.count(token) for token in overlap)
        ranked.append((score,str(row.get(id_field) or ""),row))
    ranked.sort(key=lambda item:(-item[0],item[1]))
    selected=ranked[:max(1,min(int(limit),50))]
    return [{**row,"retrieval_score":round(score,6)} for score,_identifier,row in selected]


__all__=["rank_records"]

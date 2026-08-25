"""Small public-web research surface for the autonomous engineering agent."""
from __future__ import annotations

from html import unescape
import ipaddress
import json
import re
import socket
from pathlib import Path
import hashlib
import os
from urllib.parse import quote_plus, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


_UA={"User-Agent":"Mozilla/5.0 EmbodiedCodex/1.0"}
_SEARCH_TITLE_CHARS=200
_SEARCH_SNIPPET_CHARS=600


def _text(value: str) -> str:
    value=re.sub(r"<(script|style)[^>]*>.*?</\1>"," ",value,flags=re.I|re.S)
    value=re.sub(r"<[^>]+>"," ",value)
    return " ".join(unescape(value).split())


def _bounded_search_result(item):
    """Bound untrusted provider text before it enters the Agent context."""
    result=dict(item)
    title=_text(str(result.get("title") or ""))
    snippet=_text(str(result.get("snippet") or ""))
    result["title"]=title[:_SEARCH_TITLE_CHARS]
    if len(snippet)>_SEARCH_SNIPPET_CHARS:
        result["snippet"]=snippet[:_SEARCH_SNIPPET_CHARS]+"..."
        result["snippet_truncated"]=True
    else:
        result["snippet"]=snippet
    return result


def _bing_results(html: str, limit: int):
    results=[]
    for block in re.findall(r'<li class="b_algo".*?</li>',html,flags=re.S):
        match=re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',block,flags=re.S)
        if not match: continue
        url,title=unescape(match.group(1)),_text(match.group(2))
        snippet_match=re.search(r'<p[^>]*>(.*?)</p>',block,flags=re.S)
        if url.startswith(("http://","https://")):
            results.append({"title":title,"url":url,
                            "snippet":_text(snippet_match.group(1)) if snippet_match else ""})
        if len(results)>=limit:break
    return results


def _github_results(query: str, limit: int):
    words=re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+-]*",query)
    stop={"a","an","and","for","from","in","of","on","or","the","to","using","with"}
    words=[word for word in words if word.casefold() not in stop]
    variants=[query]
    if len(words)>4:variants.append(" ".join(words[:4]))
    if len(words)>2:variants.append(" ".join(words[:2]))
    for variant in dict.fromkeys(variants):
        request=Request("https://api.github.com/search/repositories?q="+
            quote_plus(variant)+f"&per_page={limit}",headers={**_UA,
            "Accept":"application/vnd.github+json"})
        try:
            with urlopen(request,timeout=20) as response: data=json.load(response)
        except Exception:
            continue
        items=data.get("items") or []
        if items:
            return [{"title":item.get("full_name") or item.get("name"),
                "url":item.get("html_url"),"snippet":item.get("description") or "",
                "source":"github"} for item in items[:limit]]
    return []


def search_web(query: str, limit: int = 5):
    limit=max(1,min(int(limit),10)); query=str(query).strip()
    if not query: raise ValueError("query is empty")
    github=_github_results(query,limit);errors=[]
    # DuckDuckGo frequently serves a bot-interstitial with no result links.
    # Keep it as the first provider, then fall back to Bing's public HTML.
    request=Request("https://html.duckduckgo.com/html/?q="+quote_plus(query),headers=_UA)
    try:
        with urlopen(request,timeout=20) as response:
            html=response.read().decode("utf-8",errors="replace")
    except Exception as exc:
        html="";errors.append({"provider":"duckduckgo",
                               "error":f"{type(exc).__name__}: {exc}"})
    matches=re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',html,flags=re.S)
    general=[{"title":_text(title),"url":unescape(url),"snippet":"","source":"duckduckgo"}
             for url,title in matches[:limit]]
    provider="duckduckgo"
    if not general:
        request=Request("https://www.bing.com/search?q="+quote_plus(query),headers=_UA)
        try:
            with urlopen(request,timeout=20) as response:
                html=response.read().decode("utf-8",errors="replace")
            general=_bing_results(html,limit);provider="bing"
        except Exception as exc:
            general=[];provider="unavailable"
            errors.append({"provider":"bing","error":f"{type(exc).__name__}: {exc}"})
    for item in general:item.setdefault("source",provider)
    seen=set();results=[]
    for item in github+general:
        if not item.get("url") or item["url"] in seen:continue
        seen.add(item["url"]);results.append(_bounded_search_result(item))
        if len(results)>=limit:break
    return {"query":query,"provider":("github+" if github else "")+provider,
            "results":results,"provider_errors":errors}


def _public_url(url: str):
    parsed=urlparse(str(url))
    if parsed.scheme not in ("http","https") or not parsed.hostname:
        raise ValueError("only public HTTP(S) URLs are allowed")
    for item in socket.getaddrinfo(parsed.hostname,parsed.port or 443,type=socket.SOCK_STREAM):
        address=ipaddress.ip_address(item[4][0])
        if not address.is_global: raise ValueError("private or local web address is forbidden")
    return parsed.geturl()


class _PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return super().redirect_request(req,fp,code,msg,headers,_public_url(newurl))


def _open_public(url: str, *, timeout: float):
    validated=_public_url(url)
    response=build_opener(_PublicRedirectHandler()).open(Request(validated,headers=_UA),timeout=timeout)
    _public_url(response.geturl())
    return response


def fetch_web_page(url: str, max_chars: int = 30000):
    url=_public_url(url); maximum=max(1000,min(int(max_chars),100000))
    last_error=None
    for _attempt in range(2):
        try:
            with _open_public(url,timeout=25) as response:
                raw=response.read(2*1024*1024+1)
                if len(raw)>2*1024*1024: raise ValueError("web page exceeds 2 MiB")
                content_type=response.headers.get_content_type()
                charset=response.headers.get_content_charset() or "utf-8"
            break
        except ValueError:raise
        except Exception as exc:last_error=exc
    else:raise last_error
    decoded=raw.decode(charset,errors="replace")
    content=_text(decoded) if content_type in ("text/html","application/xhtml+xml") else decoded
    return {"url":url,"content_type":content_type,"content":content[:maximum],
            "truncated":len(content)>maximum}


def download_public_file(url: str,destination: str|Path,max_bytes: int=2*1024**3):
    """Stream one public HTTP(S) artifact to a pre-authorized destination."""
    target=Path(destination);maximum=max(1,min(int(max_bytes),8*1024**3))
    temporary=target.with_name(target.name+f".partial-{os.getpid()}")
    digest=hashlib.sha256();size=0
    try:
        with _open_public(url,timeout=60) as response,temporary.open("wb") as stream:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                size+=len(chunk)
                if size>maximum:raise ValueError("public asset exceeds download limit")
                digest.update(chunk);stream.write(chunk)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True);raise
    return {"url":_public_url(url),"path":str(target),"bytes":size,
            "sha256":digest.hexdigest()}


__all__=["search_web","fetch_web_page","download_public_file"]

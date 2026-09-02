"""External receipt-verifying capability promotion service (MVP trust domain)."""
from __future__ import annotations
import argparse, json, os, socketserver
from pathlib import Path
from .assets import AssetLibrary
from .control_plane import load_evidence
from .trust import verify_receipt, receipt_digest

def promote(root: Path, asset_id: str, evidence_paths: list[str], note: str, key: bytes):
    evidence = [load_evidence(p) for p in evidence_paths]
    refs = []
    for item in evidence:
        receipt = item.get("sealed_receipt")
        if not verify_receipt(receipt, key): raise ValueError("invalid evaluator receipt")
        if receipt.get("trial_id") != item.get("ref"): raise ValueError("trial binding mismatch")
        if asset_id not in item.get("assets_used", []): raise ValueError("capability was not loaded")
        refs.append(f"receipt://{receipt_digest(receipt)}")
    return AssetLibrary(root).decide_capability(asset_id, decision="promoted", evidence=refs, note=note)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--socket",type=Path,required=True)
    p.add_argument("--repository",type=Path,required=True); p.add_argument("--key-file",type=Path,required=True)
    p.add_argument("--token",default=os.getenv("ROBOFORGE_PROMOTION_TOKEN")); a=p.parse_args()
    if not a.token: p.error("service token required")
    key=a.key_file.read_bytes(); a.socket.parent.mkdir(parents=True,exist_ok=True); a.socket.unlink(missing_ok=True)
    class H(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                q=json.loads(self.rfile.readline());
                if q.get("token") != a.token: raise PermissionError("authentication failed")
                result=promote(a.repository,q["asset_id"],q["evidence"],q["note"],key)
                out={"ok":True,"result":result}
            except Exception as e: out={"ok":False,"error":f"{type(e).__name__}: {e}"}
            self.wfile.write((json.dumps(out,sort_keys=True)+"\n").encode())
    class S(socketserver.ThreadingMixIn,socketserver.UnixStreamServer): daemon_threads=True
    server=S(str(a.socket),H); os.chmod(a.socket,0o600)
    try: server.serve_forever()
    finally: server.server_close(); a.socket.unlink(missing_ok=True)
if __name__ == "__main__": main()

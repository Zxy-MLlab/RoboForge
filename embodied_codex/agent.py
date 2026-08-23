from __future__ import annotations

import json
from pathlib import Path
import time


class CodingAgent:
    def __init__(self, *, model, registry, system_prompt: str, trace_path: str|Path,
                 max_turns: int=60, model_attempts: int=3):
        self.model=model;self.registry=registry;self.system_prompt=system_prompt
        self.trace_path=Path(trace_path);self.trace_path.parent.mkdir(parents=True,exist_ok=True)
        self.max_turns=max_turns;self.model_attempts=model_attempts
    def emit(self,event):
        with self.trace_path.open("a") as f: f.write(json.dumps({"unix":time.time(),**event},default=str)+"\n")
    def run(self,instruction):
        messages=[{"role":"system","content":self.system_prompt},{"role":"user","content":instruction}]
        self.emit({"type":"task","instruction":instruction}); results=[];robot_executed=False
        for turn in range(1,self.max_turns+1):
            decision=None; error=None
            # Once a physical episode is transactionally persisted, a failed
            # post-rollout diagnosis request should end this coding pass.  The
            # next iteration can resume from the same evidence; retrying a long
            # vision request here only delays progress and risks duplicate work.
            attempts=1 if robot_executed else self.model_attempts
            for attempt in range(1,attempts+1):
                try:
                    decision=dict(self.model.decide(messages=messages,tools=self.registry.schemas));break
                except Exception as exc:
                    error=f"{type(exc).__name__}: {exc}"
                    self.emit({"type":"model_error","turn":turn,"attempt":attempt,"error":error})
                    deadline_error=(type(exc).__name__=="ModelResponseTimeout"
                                    or "model response exceeded" in str(exc))
                    if deadline_error and attempt>=2:break
                    if attempt<attempts: time.sleep(attempt*2)
            if decision is None: return {"completed":False,"error":error,"tool_results":results}
            calls=list(decision.get("tool_calls") or []);content=str(decision.get("content") or "")
            self.emit({"type":"model","turn":turn,"content":content,"tool_calls":calls})
            assistant={"role":"assistant","content":content}
            if calls: assistant["tool_calls"]=[{"id":c["id"],"type":"function",
                "function":{"name":c["name"],"arguments":c.get("arguments") or "{}"}} for c in calls]
            messages.append(assistant)
            if not calls: return {"completed":True,"final_text":content,"tool_results":results}
            pending_images=[]
            for call in calls:
                try:
                    args=json.loads(call.get("arguments") or "{}")
                    value=self.registry.invoke(call["name"],args);payload={"ok":True,"result":value}
                except Exception as exc: payload={"ok":False,"error":f"{type(exc).__name__}: {exc}"}
                results.append({"name":call["name"],**payload})
                if call["name"]=="run_robot_controller" and payload.get("ok"):
                    robot_executed=True
                image_value=(payload.get("result") or {}).get("_embodied_codex_image") \
                    if isinstance(payload.get("result"),dict) else None
                trace_payload=payload
                if isinstance(image_value,dict):
                    trace_payload={"ok":True,"result":{"image_path":image_value.get("path"),
                        "mime_type":image_value.get("mime_type"),"vision_delivered":True}}
                self.emit({"type":"tool_result","turn":turn,"name":call["name"],**trace_payload})
                messages.append({"role":"tool","tool_call_id":call["id"],
                                 "content":json.dumps(trace_payload,default=str)})
                if isinstance(image_value,dict):
                    pending_images.append({"role":"user","content":[
                        {"type":"text","text":f"Sensor image from {image_value.get('path')}. Inspect it as experimental evidence."},
                        {"type":"image_url","image_url":{"url":
                            f"data:{image_value.get('mime_type')};base64,{image_value.get('data_base64')}"}},
                    ]})
            messages.extend(pending_images)
        return {"completed":False,"error":"turn budget exhausted","tool_results":results}

__all__=["CodingAgent"]

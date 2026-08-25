from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import time


class CodingAgent:
    _PYTHON_DIAGNOSTIC_PATTERNS = (
        re.compile(r'File ["\'](?P<path>[^"\']+\.py)["\'], line (?P<line>\d+)'),
        re.compile(r'(?P<path>[A-Za-z0-9_./-]+\.py), line (?P<line>\d+)'),
        re.compile(r'(?P<path>[A-Za-z0-9_./-]+\.py):(?P<line>\d+)(?::\d+)?'),
    )
    def __init__(self, *, model, registry, system_prompt: str, trace_path: str|Path,
                 max_turns: int=60, model_attempts: int=3,
                 per_turn_tool_characters: int=50000,
                 per_turn_images: int=3,
                 max_context_characters: int=70000,
                 context_tail_messages: int=28,
                 max_evidence_deliveries: int=18,
                 max_working_memory_deliveries: int=12,
                 post_robot_evidence_deliveries: int=8,
                 post_robot_max_turns: int=20,
                 post_mutation_max_turns: int=12,
                 post_rejection_max_turns: int=8,
                 post_evidence_pause_max_turns: int=4,
                 post_duplicate_read_max_turns: int=2,
                 post_mutation_read_deliveries: int=2,
                 executable_pending: bool=False):
        self.model=model;self.registry=registry;self.system_prompt=system_prompt
        self.trace_path=Path(trace_path);self.trace_path.parent.mkdir(parents=True,exist_ok=True)
        self.max_turns=max_turns;self.model_attempts=model_attempts
        self.per_turn_tool_characters=max(10000,int(per_turn_tool_characters))
        self.per_turn_images=max(1,int(per_turn_images))
        self.max_context_characters=max(40000,int(max_context_characters))
        self.context_tail_messages=max(8,int(context_tail_messages))
        self.max_evidence_deliveries=max(6,int(max_evidence_deliveries))
        # Correction passes commonly need one plan page and two Controller
        # pages.  Do not silently raise their explicit four-page budget to six;
        # that defeated the evidence-to-action deadline in real campaigns.
        self.max_working_memory_deliveries=max(3,int(max_working_memory_deliveries))
        self.post_robot_evidence_deliveries=max(3,int(post_robot_evidence_deliveries))
        self.post_robot_max_turns=max(4,int(post_robot_max_turns))
        self.post_mutation_max_turns=max(4,int(post_mutation_max_turns))
        self.post_rejection_max_turns=max(4,int(post_rejection_max_turns))
        self.post_evidence_pause_max_turns=max(2,int(post_evidence_pause_max_turns))
        self.post_duplicate_read_max_turns=max(1,int(post_duplicate_read_max_turns))
        self.post_mutation_read_deliveries=max(0,int(post_mutation_read_deliveries))
        self.executable_pending=bool(executable_pending)
    def emit(self,event):
        with self.trace_path.open("a") as f: f.write(json.dumps({"unix":time.time(),**event},default=str)+"\n")

    @classmethod
    def _failed_command_repair_scope(cls, result):
        """Return one safe local source-read scope from a failed command receipt."""
        if not isinstance(result,dict) or result.get("exit_code") in {None,0}:
            return None
        output=str(result.get("output") or "")
        if not any(marker in output for marker in (
                "SyntaxError", "IndentationError", "TabError", "Traceback",
                "FAILED", "ERROR")):
            return None
        matches=[]
        for pattern in cls._PYTHON_DIAGNOSTIC_PATTERNS:
            matches.extend(pattern.finditer(output))
        for match in reversed(matches):
            path=Path(match.group("path"))
            if path.is_absolute() or ".." in path.parts:
                continue
            line=max(1,int(match.group("line")))
            return {"path":path.as_posix(),"line":line}
        return None

    @staticmethod
    def _is_local_repair_read(call_name, args, scope):
        if call_name!="read_file" or not isinstance(args,dict) or not scope:
            return False
        try:
            path=Path(str(args.get("path") or ""))
            start=max(1,int(args.get("start_line",1)))
            end=max(start,int(args.get("end_line",400)))
        except (TypeError,ValueError):
            return False
        if path.is_absolute() or ".." in path.parts or path.as_posix()!=scope["path"]:
            return False
        line=int(scope["line"])
        return start<=line<=end and end-start+1<=120

    def _request_audit(self,messages,tools,turn,attempt):
        encoded_messages=json.dumps(messages,sort_keys=True,separators=(",",":"),default=str)
        encoded_tools=json.dumps(tools,sort_keys=True,separators=(",",":"),default=str)
        images=[]
        for message in messages:
            if not isinstance(message.get("content"),list):continue
            for item in message["content"]:
                if isinstance(item,dict) and item.get("type")=="image_url":
                    url=str((item.get("image_url") or {}).get("url") or "")
                    images.append(hashlib.sha256(url.encode()).hexdigest())
        self.emit({"type":"model_request","turn":turn,"attempt":attempt,
            "message_count":len(messages),"message_characters":len(encoded_messages),
            "messages_sha256":hashlib.sha256(encoded_messages.encode()).hexdigest(),
            "tool_schema_sha256":hashlib.sha256(encoded_tools.encode()).hexdigest(),
            "system_prompt_sha256":hashlib.sha256(self.system_prompt.encode()).hexdigest(),
            "image_payload_sha256":images,"model":getattr(self.model,"model",None),
            "reasoning_effort":getattr(self.model,"reasoning_effort",None)})

    @staticmethod
    def _bounded_value(value, *, depth=0):
        """Keep decisive structured evidence without retaining bulk arrays."""
        if depth>=4:return "<nested evidence omitted>"
        if value is None or isinstance(value,(bool,int,float)):return value
        if isinstance(value,str):return value[:1200]
        if isinstance(value,list):
            rows=[CodingAgent._bounded_value(item,depth=depth+1) for item in value[:12]]
            if len(value)>12:rows.append({"omitted_items":len(value)-12})
            return rows
        if isinstance(value,dict):
            rows={}
            for index,(key,item) in enumerate(value.items()):
                if index>=30:
                    rows["omitted_fields"]=len(value)-30;break
                rows[str(key)]=CodingAgent._bounded_value(item,depth=depth+1)
            return rows
        return str(value)[:1200]

    @staticmethod
    def _execution_capsule(result):
        """Project the authoritative rollout decision into durable context.

        Full RPC traces remain reloadable, but context compaction must never
        erase the terminal controller result and then deny a duplicate read of
        that same evidence.  This capsule is deliberately sensor-only.
        """
        if not isinstance(result,dict) or "execution_artifact_ref" not in result:
            return {}
        report=result.get("sensor_report") or {}
        outcome=report.get("independent_task_outcome") or {}
        observations=report.get("outcome_observations") or {}
        capsule={
            "authoritative_execution_capsule":True,
            "completed":result.get("completed"),
            "error":CodingAgent._bounded_value(result.get("error")),
            "controller_result":CodingAgent._bounded_value(result.get("controller_result")),
            "sensor_success_candidate":result.get("sensor_success_candidate"),
            "transient_infrastructure_failure":CodingAgent._bounded_value(
                result.get("transient_infrastructure_failure")),
            "independent_task_outcome":CodingAgent._bounded_value(outcome),
            "outcome_observations":CodingAgent._bounded_value(observations),
            "rpc_event_count":result.get("rpc_event_count"),
            "full_execution_artifact":result.get("full_execution_artifact"),
            "execution_artifact_ref":result.get("execution_artifact_ref"),
        }
        return {key:value for key,value in capsule.items() if value is not None}

    @staticmethod
    def _compact_consumed_messages(messages, *, tool_character_limit=6000,
                                   source_budget=20_000):
        """Replace already-consumed bulk observations with reloadable receipts."""
        # Source code is active working memory, not a sensor dump. Preserve a
        # bounded set of the newest non-duplicate Python pages so the model can
        # compare and edit a controller without re-reading the same lines on
        # every turn. Everything remains reloadable after this small budget.
        preserve_source_pages=set();source_budget=max(0,int(source_budget));seen_pages=set()
        for message in reversed(messages):
            content=message.get("content")
            if message.get("role")!="tool" or not isinstance(content,str):continue
            try:payload=json.loads(content)
            except Exception:continue
            result=payload.get("result") if isinstance(payload,dict) else None
            if not isinstance(result,dict) or not isinstance(result.get("content"),str):continue
            path=str(result.get("path") or "")
            page=(path,result.get("start_line"),result.get("end_line"))
            page_characters=len(content)
            if (not path.endswith(".py") or page in seen_pages
                    or page_characters>20_000 or page_characters>source_budget):continue
            preserve_source_pages.add(id(message));seen_pages.add(page)
            source_budget-=page_characters
        for message in messages:
            content=message.get("content")
            if message.get("role")=="user" and isinstance(content,list):
                text_parts=[str(item.get("text") or "") for item in content
                            if isinstance(item,dict) and item.get("type")=="text"]
                if any(isinstance(item,dict) and item.get("type")=="image_url"
                       for item in content):
                    message["content"]=(" ".join(text_parts).strip()+
                        " [Image bytes consumed; call view_sensor_image on the recorded path to inspect again.]")
            elif (message.get("role")=="tool" and isinstance(content,str)
                  and id(message) not in preserve_source_pages
                  and len(content)>tool_character_limit):
                try:payload=json.loads(content)
                except Exception:
                    payload={"ok":False,"compacted_text_prefix":content[:1000]}
                result=payload.get("result") if isinstance(payload,dict) else None
                receipt={"ok":payload.get("ok") if isinstance(payload,dict) else None,
                         "working_memory_compacted":True}
                if isinstance(result,dict):
                    for key in ("path","image_path","rollout_path","full_execution_artifact",
                                "execution_artifact_ref","start_line","end_line","total_lines",
                                "next_start_line","content_truncated","tool_id","skill_id",
                                "experience_id","gap_id","status","exit_code","timed_out"):
                        if key in result:receipt[key]=result[key]
                    receipt.update(CodingAgent._execution_capsule(result))
                    receipt["result_keys"]=sorted(str(key) for key in result)[:80]
                elif isinstance(payload,dict) and payload.get("error"):
                    receipt["error"]=str(payload["error"])[:2000]
                message["content"]=json.dumps(receipt,default=str)

    @staticmethod
    def _deferred_payload(payload,reason):
        result=payload.get("result") if isinstance(payload,dict) else None
        receipt={"ok":payload.get("ok") if isinstance(payload,dict) else None,
                 "delivery_deferred":True,"reason":reason,
                 "instruction":"Request the specific path/page/image again in the next turn."}
        if isinstance(result,dict):
            for key in ("path","image_path","rollout_path","full_execution_artifact",
                        "execution_artifact_ref","start_line","end_line","total_lines",
                        "next_start_line","content_truncated","mime_type"):
                if key in result:receipt[key]=result[key]
            receipt["result_keys"]=sorted(str(key) for key in result)[:80]
        return receipt

    @staticmethod
    def _read_signature(name, arguments):
        normalized=json.dumps(arguments,sort_keys=True,separators=(",",":"),
                              ensure_ascii=True,default=str)
        return hashlib.sha256(f"{name}\n{normalized}".encode()).hexdigest()

    @staticmethod
    def _asset_search_fingerprint(name, result):
        """Identify asset searches that returned the same capability set.

        Retrieval queries are free-form, so exact argument hashing treats
        paraphrases as new evidence even when they return precisely the same
        Tools, Skills, Experiences, and Gaps.  The IDs are the authoritative
        immutable result; ranking scores and query wording are not new
        evidence.  Empty searches are deliberately not fingerprinted because
        a different query may still be a useful acquisition attempt.
        """
        if name!="search_assets" or not isinstance(result,dict):return None
        assets=[]
        for collection in ("tools","skills","experiences","gaps","models",
                           "algorithms","packages"):
            rows=result.get(collection)
            if not isinstance(rows,list):continue
            for row in rows:
                if not isinstance(row,dict):continue
                identifier=next((row.get(key) for key in
                    ("tool_id","skill_id","experience_id","gap_id","model_id",
                     "algorithm_id","package_id","asset_id") if row.get(key)),None)
                if identifier is not None:assets.append((collection,str(identifier)))
        if not assets:return None
        normalized=json.dumps(sorted(set(assets)),separators=(",",":"))
        return hashlib.sha256(normalized.encode()).hexdigest(),sorted(set(assets))

    @staticmethod
    def _evidence_receipt(name, arguments, payload, deliveries, epoch):
        result=payload.get("result") if isinstance(payload,dict) else None
        receipt={"ok":True,"duplicate_read_suppressed":True,
                 "tool":name,"arguments":arguments,
                 "prior_successful_deliveries":deliveries,
                 "evidence_epoch":epoch,
                 "instruction":("This exact evidence was already delivered in the current "
                    "evidence epoch. Act on it or request different evidence.")}
        if isinstance(result,dict):
            for key in ("path","image_path","rollout_path","full_execution_artifact",
                        "execution_artifact_ref","start_line","end_line","total_lines",
                        "next_start_line","tool_id","skill_id","experience_id","gap_id",
                        "status","mime_type"):
                if key in result:receipt[key]=result[key]
            receipt["result_keys"]=sorted(str(key) for key in result)[:80]
        return receipt

    def _evidence_budget_receipt(self, name, arguments, deliveries, limit):
        return {"ok":True,"evidence_acquisition_paused":True,"tool":name,
                "arguments":arguments,"delivered_in_current_diagnosis":deliveries,
                "delivery_limit":limit,
                "instruction":("Stop varying read parameters. Externalize the current evidence-backed "
                    "hypotheses in a workspace note, modify/test the Controller, register or revise an "
                    "evidence-backed asset, or run the Controller. A substantive engineering action "
                    "opens a new bounded evidence phase.")}

    @staticmethod
    def _working_memory_budget_receipt(name, arguments, deliveries, limit):
        return {"ok":True,"working_memory_acquisition_paused":True,"tool":name,
                "arguments":arguments,"delivered_in_current_engineering_phase":deliveries,
                "delivery_limit":limit,
                "instruction":("Enough workspace pages have been delivered. Modify/test the "
                    "Controller, update an executable asset, or run the Controller now. A "
                    "substantive engineering mutation opens a new bounded workspace-read phase.")}

    def _compact_context_window(self, messages, *, latest_execution_available=False):
        """Bound accumulated tool chatter while keeping a valid recent turn.

        File and execution artifacts are immutable and reloadable through the
        registry, so retaining every historical page in the API context is
        unnecessary.  Keep the task prompt and the most recent complete tool
        interaction window; replace older interactions with a short receipt.
        The replacement is a user message, avoiding orphaned assistant
        ``tool_calls`` that some API implementations reject.
        """
        encoded=json.dumps(messages,default=str,separators=(",",":"))
        if len(encoded)<=self.max_context_characters:return messages
        head=messages[:2]
        desired=max(2,len(messages)-self.context_tail_messages)
        def safe_boundary(index):
            message=messages[index]
            if (message.get("role")=="user" or
                    (message.get("role")=="assistant" and not message.get("tool_calls"))):
                return True
            # The newest assistant -> Tool chain contains results which the
            # model has not consumed yet.  Starting a suffix at the assistant
            # is legal when the complete linked chain is present; excluding
            # this boundary used to drop every freshly requested result before
            # the next model request whenever that chain crossed the context
            # limit.  Older complete chains remain eligible for ordinary tail
            # compaction as well.
            return (message.get("role")=="assistant"
                    and bool(message.get("tool_calls"))
                    and self._tool_call_links_valid(messages[index:]))
        candidates=[index for index in range(desired,len(messages)) if safe_boundary(index)]
        # Retain the newest complete assistant -> Tool chain when possible so
        # freshly requested evidence is delivered exactly once.  The boundary
        # test above prevents orphaned function outputs.
        start=candidates[0] if candidates else len(messages)
        dropped=start-len(head)
        receipt={"working_memory_compacted":True,
                 "dropped_message_count":dropped,
                 "reason":"Earlier Tool pages and model chatter remain reloadable from immutable run artifacts.",
                 "instruction":"Use the current Controller and indexed execution evidence; reload only the specific page needed."}
        if latest_execution_available:
            receipt.update({
                "robot_executed_in_current_pass":True,
                "authoritative_current_execution":"latest_robot_execution",
                "authoritative_current_rollout":"latest_rollout",
                "previous_alias_scope":"previous_robot_execution and previous_rollout refer to the prior episode only",
                "instruction":("Diagnose the robot episode just executed in this pass from "
                    "latest_robot_execution/latest_rollout. Use previous_* only for an explicit "
                    "historical comparison; then persist the diagnosis before ending the pass.")})
        compacted=head+[{"role":"user","content":json.dumps(receipt)}]+messages[start:]
        # A single unusually large recent response should still be bounded.
        later=[index for index in range(start+1,len(messages)) if safe_boundary(index)]
        while (len(json.dumps(compacted,default=str,separators=(",",":")))>
               self.max_context_characters and later):
            start=later.pop(0);receipt["dropped_message_count"]=start-len(head)
            compacted=head+[{"role":"user","content":json.dumps(receipt)}]+messages[start:]
        # A recent complete assistant -> Tool chain may itself be larger than
        # the window. Compact its structured payloads in place before falling
        # back to a reload receipt. Never send a best-effort over-limit request
        # to the model.
        if (len(json.dumps(compacted,default=str,separators=(",",":")))>
                self.max_context_characters):
            self._compact_consumed_messages(compacted,tool_character_limit=2500)
        # If the task prompt plus a full source page still exceeds the hard
        # window, keep the complete linked Tool chain but turn that oversized
        # page into an addressable receipt.  The model can request a narrower
        # line range next turn.  Dropping the entire chain here made *all*
        # freshly requested manuals/evidence invisible and caused read loops.
        if (len(json.dumps(compacted,default=str,separators=(",",":")))>
                self.max_context_characters):
            self._compact_consumed_messages(compacted,tool_character_limit=2500,
                                            source_budget=0)
        if (len(json.dumps(compacted,default=str,separators=(",",":")))>
                self.max_context_characters):
            compacted=head+[{"role":"user","content":json.dumps({**receipt,
                "reason":"The newest complete Tool chain exceeded the context window and remains reloadable from immutable artifacts."})}]
        if (len(json.dumps(compacted,default=str,separators=(",",":")))>
                self.max_context_characters):
            raise RuntimeError("task and system prompts exceed the configured context character limit")
        return compacted

    @staticmethod
    def _tool_call_links_valid(messages):
        pending=set()
        for message in messages:
            role=message.get("role")
            if role=="assistant":
                if pending:return False
                pending={str(call.get("id")) for call in (message.get("tool_calls") or [])}
            elif role=="tool":
                call_id=str(message.get("tool_call_id"))
                if call_id not in pending:return False
                pending.remove(call_id)
            elif pending:
                return False
        return not pending
    def run(self,instruction):
        messages=[{"role":"system","content":self.system_prompt},{"role":"user","content":instruction}]
        self.emit({"type":"task","instruction":instruction}); results=[];robot_executed=False
        delivered_reads={};evidence_epochs={};evidence_deliveries=0
        delivered_asset_searches={}
        working_memory_deliveries=0
        robot_executed_turn=None
        unchanged_rejection_turn=None
        evidence_pause_turn=None
        duplicate_read_turn=None
        executable_mutation_turn=0 if self.executable_pending else None
        post_mutation_reads=0
        repair_read_scope=None
        for turn in range(1,self.max_turns+1):
            if (robot_executed_turn is not None
                    and turn>robot_executed_turn+self.post_robot_max_turns):
                return {"completed":False,
                        "error":"post-robot diagnosis turn budget exhausted",
                        "tool_results":results}
            if (robot_executed_turn is None and executable_mutation_turn is not None
                    and turn>executable_mutation_turn+self.post_mutation_max_turns):
                return {"completed":False,
                        "error":"controller execution deadline after executable mutation",
                        "tool_results":results}
            if (unchanged_rejection_turn is not None
                    and turn>unchanged_rejection_turn+self.post_rejection_max_turns):
                return {"completed":False,
                        "error":"controller mutation deadline after unchanged replay rejection",
                        "tool_results":results}
            if (evidence_pause_turn is not None
                    and turn>evidence_pause_turn+self.post_evidence_pause_max_turns):
                return {"completed":False,
                        "error":"engineering action deadline after evidence pause",
                        "tool_results":results}
            if (duplicate_read_turn is not None
                    and turn>duplicate_read_turn+self.post_duplicate_read_max_turns):
                return {"completed":False,
                        "error":"engineering action deadline after repeated evidence request",
                        "tool_results":results}
            decision=None; error=None
            # Once a physical episode is transactionally persisted, a failed
            # post-rollout diagnosis request should end this coding pass.  The
            # next iteration can resume from the same evidence; retrying a long
            # vision request here only delays progress and risks duplicate work.
            attempts=1 if robot_executed else self.model_attempts
            for attempt in range(1,attempts+1):
                try:
                    schemas=self.registry.schemas
                    messages=self._compact_context_window(
                        messages,latest_execution_available=robot_executed)
                    if not self._tool_call_links_valid(messages):
                        raise RuntimeError("internal context compaction produced invalid Tool call links")
                    self._request_audit(messages,schemas,turn,attempt)
                    decision=dict(self.model.decide(messages=messages,tools=schemas));break
                except Exception as exc:
                    error=f"{type(exc).__name__}: {exc}"
                    self.emit({"type":"model_error","turn":turn,"attempt":attempt,"error":error})
                    deadline_error=(type(exc).__name__=="ModelResponseTimeout"
                                    or "model response exceeded" in str(exc))
                    # A 120 s full-response deadline already consumed a large
                    # slice of an experiment; permit one retry in this context,
                    # then let the outer lifecycle resume with a fresh prompt.
                    if deadline_error and attempt>=2:break
                    if attempt<attempts: time.sleep(attempt*2)
            if decision is None: return {"completed":False,"error":error,"tool_results":results}
            # The current context has just been consumed. Keep reloadable
            # receipts, not repeated base64 frames and multi-page logs.
            self._compact_consumed_messages(messages)
            calls=list(decision.get("tool_calls") or []);content=str(decision.get("content") or "")
            self.emit({"type":"model","turn":turn,"content":content,"tool_calls":calls})
            assistant={"role":"assistant","content":content}
            if calls: assistant["tool_calls"]=[{"id":c["id"],"type":"function",
                "function":{"name":c["name"],"arguments":c.get("arguments") or "{}"}} for c in calls]
            messages.append(assistant)
            if not calls: return {"completed":True,"final_text":content,"tool_results":results}
            pending_images=[];deferred_image_notes=[];turn_tool_characters=0
            turn_duplicate_read=False;turn_new_evidence=False
            turn_engineering_progress=False
            for call in calls:
                args={};policy="repeatable";group="default";invalidates=();progress=False
                execution_progress=False;repair_read=False
                post_mutation_read_allowed=False
                signature=None;cached=None;asset_search_fingerprint=None
                try:
                    args=json.loads(call.get("arguments") or "{}")
                    contract=self.registry.evidence_contract(call["name"],args)
                    policy=contract["policy"];group=contract["group"]
                    invalidates=contract["invalidates"];progress=contract["progress"]
                    execution_progress=contract["execution_progress"]
                    post_mutation_read_allowed=contract[
                        "post_mutation_read_allowed"]
                    pending_read=(robot_executed_turn is None
                                  and executable_mutation_turn is not None
                                  and policy in {"read_once","working_memory",
                                                 "image_twice","budgeted_output"}
                                  and call["name"]!="run_command")
                    repair_read=(pending_read and
                        self._is_local_repair_read(call["name"],args,repair_read_scope))
                    if policy in {"read_once","working_memory","image_twice",
                                  "budgeted_output"}:
                        signature=self._read_signature(call["name"],args)
                        cached=delivered_reads.get(signature)
                    limit=2 if policy=="image_twice" else 1
                    evidence_limit=(self.post_robot_evidence_deliveries
                                    if robot_executed else self.max_evidence_deliveries)
                    if (pending_read and not repair_read
                            and not post_mutation_read_allowed and
                            post_mutation_reads>=self.post_mutation_read_deliveries):
                        payload={"ok":False,"evidence_acquisition_paused":True,
                            "controller_execution_pending":True,
                            "tool":call["name"],"arguments":args,
                            "delivered_after_mutation":post_mutation_reads,
                            "delivery_limit":self.post_mutation_read_deliveries,
                            "instruction":("The Controller already has unexecuted semantic "
                                "changes. Stop inspecting prior evidence. Compile/test, correct "
                                "a concrete error if one is reported, or call "
                                "run_robot_controller now.")}
                    elif cached is not None and cached["deliveries"]>=limit:
                        payload=self._evidence_receipt(call["name"],args,
                                                       cached["payload"],cached["deliveries"],
                                                       evidence_epochs.get(group,0))
                    elif (policy=="working_memory" and
                          working_memory_deliveries>=self.max_working_memory_deliveries):
                        payload=self._working_memory_budget_receipt(
                            call["name"],args,working_memory_deliveries,
                            self.max_working_memory_deliveries)
                    elif (policy in {"read_once","image_twice"}
                          and evidence_deliveries>=evidence_limit):
                        payload=self._evidence_budget_receipt(
                            call["name"],args,evidence_deliveries,evidence_limit)
                    else:
                        value=self.registry.invoke(call["name"],args)
                        payload={"ok":True,"result":value}
                        fingerprint=self._asset_search_fingerprint(call["name"],value)
                        if fingerprint is not None:
                            asset_search_fingerprint,asset_ids=fingerprint
                            prior_search=delivered_asset_searches.get(asset_search_fingerprint)
                            if prior_search is not None:
                                payload={"ok":True,"duplicate_read_suppressed":True,
                                    "semantic_duplicate_suppressed":True,
                                    "tool":call["name"],"arguments":args,
                                    "prior_arguments":prior_search["arguments"],
                                    "returned_asset_ids":asset_ids,
                                    "instruction":("This paraphrased search returned the same "
                                        "immutable asset set already delivered. Inspect a selected "
                                        "asset, acquire/register a missing capability, modify/test "
                                        "the Controller, or run the experiment.")}
                except Exception as exc: payload={"ok":False,"error":f"{type(exc).__name__}: {exc}"}
                result_value=payload.get("result") if isinstance(payload,dict) else None
                dynamic_workspace_mutation=bool(isinstance(result_value,dict) and
                    result_value.get("_embodied_codex_engineering_progress"))
                semantic_progress=(result_value.get("_embodied_codex_semantic_progress")
                    if isinstance(result_value,dict) else None)
                if semantic_progress is False:
                    progress=False;execution_progress=False
                if dynamic_workspace_mutation:
                    progress=True
                    if result_value.get("_embodied_codex_controller_mutated"):
                        execution_progress=True
                if (pending_read and payload.get("ok")
                        and not payload.get("duplicate_read_suppressed")):
                    post_mutation_reads+=1
                    if repair_read:
                        repair_read_scope=None
                if call["name"]=="run_command" and payload.get("ok"):
                    repair_read_scope=self._failed_command_repair_scope(result_value)
                if call["name"]=="run_robot_controller" and payload.get("ok"):
                    robot_executed=True
                    if robot_executed_turn is None:robot_executed_turn=turn
                elif call["name"]=="run_robot_controller" and not payload.get("ok"):
                    rejection=str(payload.get("error") or "")
                    if ("unchanged_controller_after_failed_episode" in rejection or
                            "repeated_strategy_after_failed_episodes" in rejection):
                        unchanged_rejection_turn=turn
                image_value=(payload.get("result") or {}).get("_embodied_codex_image") \
                    if isinstance(payload.get("result"),dict) else None
                trace_payload=payload
                image_will_be_delivered=(isinstance(image_value,dict)
                                         and len(pending_images)<self.per_turn_images)
                if isinstance(image_value,dict):
                    trace_payload={"ok":True,"result":{"image_path":image_value.get("path"),
                        "mime_type":image_value.get("mime_type"),
                        "vision_delivered":image_will_be_delivered}}
                model_payload=(trace_payload if not isinstance(image_value,dict)
                    or image_will_be_delivered else self._deferred_payload(
                        trace_payload,"per-turn image budget reached"))
                encoded=json.dumps(model_payload,default=str)
                budgeted_evidence=(policy=="budgeted_output" and len(encoded)>2000
                                   and not payload.get("duplicate_read_suppressed"))
                evidence_limit=(self.post_robot_evidence_deliveries
                                if robot_executed else self.max_evidence_deliveries)
                if (budgeted_evidence
                        and evidence_deliveries>=evidence_limit):
                    model_payload=self._evidence_budget_receipt(
                        call["name"],args,evidence_deliveries,evidence_limit)
                    encoded=json.dumps(model_payload,default=str)
                if (turn_tool_characters+len(encoded)>self.per_turn_tool_characters
                        and len(encoded)>6000):
                    model_payload=self._deferred_payload(
                        trace_payload,"per-turn structured evidence budget reached")
                    encoded=json.dumps(model_payload,default=str)
                turn_tool_characters+=len(encoded)
                duplicate=bool(payload.get("duplicate_read_suppressed"))
                if duplicate:turn_duplicate_read=True
                if (execution_progress and payload.get("ok") and not duplicate
                        and executable_mutation_turn is None):
                    executable_mutation_turn=turn
                    post_mutation_reads=0
                if execution_progress and payload.get("ok") and not duplicate:
                    unchanged_rejection_turn=None
                acquisition_paused=bool(model_payload.get("evidence_acquisition_paused")
                    or model_payload.get("working_memory_acquisition_paused"))
                if acquisition_paused and evidence_pause_turn is None:
                    evidence_pause_turn=turn
                actually_delivered=(payload.get("ok") and not duplicate and not acquisition_paused
                                    and not model_payload.get("delivery_deferred")
                                    and (not isinstance(image_value,dict)
                                     or image_will_be_delivered))
                if actually_delivered and (policy in {"read_once","working_memory",
                                                       "image_twice"}
                                           or budgeted_evidence):
                    turn_new_evidence=True
                evidence_bearing=(policy in {"read_once","image_twice"}
                                  or budgeted_evidence)
                cacheable=(policy in {"read_once","working_memory","image_twice"}
                           or budgeted_evidence)
                if signature is not None and actually_delivered and cacheable:
                    prior=delivered_reads.get(signature)
                    delivered_reads[signature]={"deliveries":
                        (prior["deliveries"] if prior else 0)+1,
                        "payload":trace_payload,"group":group,
                        "epoch":evidence_epochs.get(group,0)}
                    if evidence_bearing:evidence_deliveries+=1
                    if policy=="working_memory":working_memory_deliveries+=1
                if (asset_search_fingerprint is not None and actually_delivered
                        and not duplicate):
                    delivered_asset_searches[asset_search_fingerprint]={
                        "arguments":dict(args)}
                results.append({"name":call["name"],**model_payload})
                self.emit({"type":"tool_result","turn":turn,"name":call["name"],**trace_payload})
                messages.append({"role":"tool","tool_call_id":call["id"],
                                 "content":json.dumps(model_payload,default=str)})
                if isinstance(image_value,dict):
                    if image_will_be_delivered:
                        pending_images.append({"role":"user","content":[
                            {"type":"text","text":f"Sensor image from {image_value.get('path')}. Inspect it as experimental evidence."},
                            {"type":"image_url","image_url":{"url":
                                f"data:{image_value.get('mime_type')};base64,{image_value.get('data_base64')}"}},
                        ]})
                    else:
                        deferred_image_notes.append(
                            f"Sensor image delivery deferred for {image_value.get('path')} because the per-turn image budget was reached. Request it again next turn if needed.")
                if ((policy=="invalidates_reads" or dynamic_workspace_mutation)
                        and payload.get("ok") and not duplicate):
                    groups=set(invalidates or
                               (("workspace",) if dynamic_workspace_mutation else ("default",)))
                    delivered_reads={key:value for key,value in delivered_reads.items()
                                     if value.get("group") not in groups}
                    for invalidated_group in groups:
                        evidence_epochs[invalidated_group]=(
                            evidence_epochs.get(invalidated_group,0)+1)
                if progress and payload.get("ok") and not duplicate:
                    turn_engineering_progress=True
                    evidence_deliveries=0
                    working_memory_deliveries=0
                    evidence_pause_turn=None
            # A duplicate receipt is useful once: it tells the model that the
            # requested evidence is already in working memory.  Repeating the
            # same request is not diagnosis.  Bound that stagnation separately
            # from the ordinary evidence budget, while allowing a genuinely new
            # observation or any engineering mutation to reopen the loop.
            if turn_engineering_progress or turn_new_evidence:
                duplicate_read_turn=None
            elif turn_duplicate_read and duplicate_read_turn is None:
                duplicate_read_turn=turn
            messages.extend(pending_images)
            if deferred_image_notes:
                messages.append({"role":"user","content":"\n".join(deferred_image_notes)})
            if any(row.get("name")=="run_robot_controller" and row.get("ok")
                   and isinstance(row.get("result"),dict)
                   and row["result"].get("sensor_success_candidate") is True
                   for row in results):
                return {"completed":True,"final_text":"sensor success; proceed to locked validation",
                        "tool_results":results}
        return {"completed":False,"error":"turn budget exhausted","tool_results":results}

__all__=["CodingAgent"]

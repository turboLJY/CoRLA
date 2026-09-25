"""CoRLA proxy: collect complete trajectories and query two distinct teachers.

The fixed PRM supplies labels/hints. The current frozen rollout policy supplies
both original-context and hint-context sampled-token probabilities.
"""

from __future__ import annotations

import asyncio
import logging
import math

import httpx
from fastapi import Header, HTTPException, Request

from openclaw_opd_api_server import (
    OpenClawOPDAPIServer, _append_hint_to_messages, _build_hint_judge_messages,
    _flatten_message_content, _normalize_messages_for_template,
)

logger = logging.getLogger(__name__)


def parse_sampled_log_probs(result, response_ids):
    """Reject missing/misaligned probabilities instead of substituting zeros."""
    pairs = result.get("meta_info", {}).get("input_token_logprobs")
    n = len(response_ids)
    if not n or not isinstance(pairs, list) or len(pairs) < n:
        raise ValueError("rollout-policy scorer omitted sampled-token log probabilities")
    tail = pairs[-n:]
    if any(not isinstance(p, (list, tuple)) or len(p) < 2 for p in tail):
        raise ValueError("invalid sampled-token probability format")
    if [p[1] for p in tail] != list(response_ids):
        raise ValueError("scored token IDs differ from the sampled action")
    values = [p[0] for p in tail]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v > 1e-5 for v in values):
        raise ValueError("invalid sampled-token log probability")
    return [float(v) for v in values]


class UALCAAPIServer(OpenClawOPDAPIServer):
    def __init__(self, args, output_queue, submission_enabled):
        self._session_locks = {}
        self._round_sessions = set()
        self._finished_sessions = set()
        self._round_limit = args.rollout_batch_size
        self._round_id = 0
        self._loop = None
        self._query_semaphore = asyncio.Semaphore(4)
        super().__init__(args, output_queue, submission_enabled)
        if not self._prm_enabled or not self._prm_url:
            raise ValueError("CoRLA requires a frozen PRM for labels and hints")
        self._prm_m = 1
        self._prm_temperature = 0.0
        self._policy_score_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        self._max_context = args.rollout_max_context_len

    def _build_app(self):
        app = super()._build_app()

        @app.on_event("startup")
        async def remember_loop():
            self._loop = asyncio.get_running_loop()

        @app.post("/v1/ualca/feedback")
        async def feedback(request: Request, authorization: str | None = Header(default=None)):
            await self._check_auth(authorization)
            if not self.submission_enabled.is_set():
                raise HTTPException(503, "submission paused for policy update")
            return await self.finish_trajectory(await request.json())

        return app

    def begin_round(self, round_id):
        if self._pending_turn_data:
            raise RuntimeError("cannot change policy while a trajectory is still open")
        self._round_id = round_id
        self._round_sessions.clear()
        self._finished_sessions.clear()
        self._session_locks.clear()
        self.submission_enabled.set()

    def _encode_messages(self, messages, tools=None):
        normalized = _normalize_messages_for_template(messages)
        text = self.tokenizer.apply_chat_template(
            normalized, tools=tools, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self._max_context:
            raise HTTPException(400, "full history exceeds the CoRLA context limit")
        return ids

    async def _handle_request(self, body, session_id, turn_type, session_done):
        if session_done or "terminal_reward" in body:
            raise HTTPException(400, "send terminal_reward and final messages to /v1/ualca/feedback after the final action")
        if session_id == "unknown":
            raise HTTPException(400, "CoRLA requires a distinct session_id per trajectory")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must contain the complete interaction history")
        max_tokens = body.get("max_tokens", 4096)
        if type(max_tokens) is not int or max_tokens < 1:
            raise HTTPException(400, "max_tokens must be a positive integer")
        # The scorer and policy objective use the untempered policy. Keep the
        # sampling distribution identical even if a client has older defaults.
        body = {**body, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "max_tokens": min(max_tokens, 4096)}
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            if not self.submission_enabled.is_set():
                raise HTTPException(503, "submission paused for policy update")
            if session_id in self._finished_sessions:
                raise HTTPException(409, "trajectory already finished in this round")
            is_new = session_id not in self._round_sessions
            if turn_type == "main" and is_new:
                if len(self._round_sessions) >= self._round_limit:
                    raise HTTPException(503, "round is full; finish existing sessions, then retry after the policy update")
                self._round_sessions.add(session_id)
            pending = self._pending_turn_data.get(session_id, {})
            if turn_type == "main" and pending:
                previous = pending[max(pending)]
                previous["transition_ids"] = self._encode_messages(body["messages"], body.get("tools"))
            try:
                result = await super()._handle_request(body, session_id, turn_type, False)
                if is_new and not self._pending_turn_data.get(session_id):
                    self._round_sessions.discard(session_id)
                return result
            except Exception:
                if is_new and not self._pending_turn_data.get(session_id):
                    self._round_sessions.discard(session_id)
                raise

    def _fire_opd_task(self, session_id, turn_num, turn_data, next_state):
        # Teacher calls are selected by sigma only after the complete batch is
        # collected. Completion order cannot reorder trajectory transitions.
        turn_data["next_state"] = next_state
        turn_data["has_next_state"] = True

    def _maybe_submit_ready_samples(self, session_id, force_drop_without_next_state=False):
        # No short-lookahead flush: terminal returns supervise full histories.
        return

    async def finish_trajectory(self, body):
        session_id = body.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise HTTPException(400, "session_id is required")
        terminated = body.get("terminated", True)
        if type(terminated) is not bool:
            raise HTTPException(400, "terminated must be a JSON boolean")
        outcome = body.get("terminal_reward")
        if terminated:
            if type(outcome) not in (int, float) or not math.isfinite(outcome) or not 0 <= outcome <= 1:
                raise HTTPException(400, "true termination requires verified terminal_reward in [0,1]")
        elif outcome is not None:
            raise HTTPException(400, "a truncation must omit terminal_reward")
        messages = body.get("messages")
        if (not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict)
                or messages[-1].get("role") not in ("user", "tool")):
            raise HTTPException(400, "messages must contain the complete history ending in final user/tool feedback")
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            if session_id in self._finished_sessions:
                return {"session_id": session_id, "status": "already_finished", "round": self._round_id}
            pending = self._pending_turn_data.get(session_id)
            if not pending:
                raise HTTPException(404, "no pending actions for this session")
            turns = [pending[k] for k in sorted(pending)]
            final = turns[-1]
            final["transition_ids"] = self._encode_messages(messages, final.get("tools"))
            final["next_state"] = messages[-1]
            final["has_next_state"] = True
            if any(not t.get("transition_ids") or not t.get("has_next_state") for t in turns):
                raise HTTPException(409, "a trajectory transition is missing its next-state feedback")
            trajectory = {"session_id": session_id, "round": self._round_id,
                          "transitions": turns,
                          "states": [t["prompt_ids"] for t in turns] + [final["transition_ids"]],
                          "terminated": terminated, "outcome": outcome}
            # Queue is unbounded and one item is one complete trajectory.
            self.output_queue.put_nowait(trajectory)
            self._finished_sessions.add(session_id)
            self._pending_turn_data.pop(session_id, None)
            self._turn_counts.pop(session_id, None)
            self._flush_pending_record(session_id, messages[-1])
            return {"session_id": session_id, "status": "finished", "round": self._round_id, "turns": len(turns)}

    def _teacher_prompt(self, transition, hint=False):
        next_state = transition["next_state"]
        observation = _flatten_message_content(next_state.get("content"))
        if hint:
            messages = _build_hint_judge_messages(transition["response_text"], observation,
                                                   next_state.get("role", "user"))
        else:
            messages = [
                {"role": "system", "content": (
                    "Assess signed local progress toward the overall task using the history, action, and next observation. "
                    "Use +1 for clear useful progress, -1 for clear harmful or misdirected progress, and 0 for "
                    "no demonstrated progress or ambiguous evidence. A failed command may yield useful diagnostic "
                    "information; a successful command alone does not establish progress. This is a noisy process "
                    "label, not a verified terminal outcome. Return only \\boxed{1}, \\boxed{0}, or \\boxed{-1}." )},
                {"role": "user", "content": f"## Action\n{transition['response_text']}\n\n## Next observation\n{observation}"},
            ]
        # The paper's x_t includes h_t. The legacy PRM prompt only contained
        # the action and observation, so add the complete preceding history.
        messages[-1]["content"] = "## History before the action\n" + transition["prompt_text"] + "\n\n" + messages[-1]["content"]
        tokenizer = self._prm_tokenizer
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    async def query_teacher_label(self, transition):
        async with self._query_semaphore:
            return await self._query_prm_eval_once(self._teacher_prompt(transition), 0)

    async def query_hint(self, transition):
        async with self._query_semaphore:
            vote = await self._query_judge_once(self._teacher_prompt(transition, hint=True), 0)
        hint = vote.get("hint")
        return hint.strip() if vote.get("score") == 1 and isinstance(hint, str) and hint.strip() else None

    async def score_policy(self, prompt_ids, response_ids):
        ids = list(prompt_ids) + list(response_ids)
        if not prompt_ids or len(ids) > self._max_context:
            raise ValueError("OPD scoring context is empty or exceeds context limit")
        payload = {"input_ids": ids,
                   "sampling_params": {"temperature": 1.0, "max_new_tokens": 0, "skip_special_tokens": False},
                   "return_logprob": True, "logprob_start_len": len(prompt_ids) - 1}
        async with self._teacher_lp_semaphore:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(self._policy_score_url, json=payload)
                response.raise_for_status()
                return parse_sampled_log_probs(response.json(), response_ids)

    def hinted_prompt_ids(self, transition, hint):
        messages = _append_hint_to_messages(transition["messages"], hint)
        return self._encode_messages(messages, transition.get("tools"))

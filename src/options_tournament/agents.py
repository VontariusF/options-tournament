"""Featherless specialist roster that proposes tournament strategy cards.

Python owns structure, delta, DTE, and wing buckets. Models propose and critique
in that discrete space; they never emit risk, premium, or payoff numbers.

Roster (env-overridable):
  * hypothesis  — OpenMath-Nemotron-32B : complementary underlyings and theses
  * strategist  — DeepSeek-V3-0324      : structure / DTE / delta under the book mix
  * proposer    — Qwen3-32B             : investigates via read-only tools, submits cards
  * critic      — DeepSeek-V3-0324      : novelty and venue flags only
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from options_tournament.execute import STRUCTURES, StrategyCard
from options_tournament.featherless import FeatherlessClient, LLMError

HYPOTHESIS_MODEL = os.environ.get("DISCOVERY_HYPOTHESIS_MODEL", "nvidia/OpenMath-Nemotron-32B")
STRATEGIST_MODEL = os.environ.get("DISCOVERY_STRATEGIST_MODEL", "deepseek-ai/DeepSeek-V3-0324")
PROPOSER_MODEL = os.environ.get("DISCOVERY_PROPOSER_MODEL", "Qwen/Qwen3-32B")
CRITIC_MODEL = os.environ.get("DISCOVERY_CRITIC_MODEL", "deepseek-ai/DeepSeek-V3-0324")

MAX_TOOL_TURNS = int(os.environ.get("DISCOVERY_MAX_TOOL_TURNS", "6") or 6)

ALLOWED_DTE = (7, 14, 30)
ALLOWED_DELTA = (0.20, 0.35, 0.50)
ALLOWED_WING = (5.0, 10.0, 25.0)
DEFAULT_UNIVERSE = ("AAPL", "MSFT", "NVDA", "SPY", "QQQ")

_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "underlying": {"type": "string"},
        "structure": {"type": "string", "enum": list(STRUCTURES)},
        "dte": {"type": "integer", "enum": list(ALLOWED_DTE)},
        "delta": {"type": "number", "enum": list(ALLOWED_DELTA)},
        "wing_width": {"type": ["number", "null"]},
        "thesis": {"type": "string"},
    },
    "required": ["underlying", "structure"],
}

_VENUE = (
    "Venue (Alpaca paper, only submittable): US equity/ETF OCC options. Structures: long_call, "
    "long_put (single), credit_put_spread (2-leg defined-risk). No index, crypto, naked short, "
    "or iron condor. dte in {7,14,30}; delta in {0.20,0.35,0.50}; wing_width in {5,10,25} for "
    "spreads only (omit otherwise). Underlying MUST come from the universe. NEVER emit risk, "
    "max-loss, premium, or payoff numbers."
)


def _nearest(value: float, allowed: tuple) -> float:
    return min(allowed, key=lambda x: abs(float(x) - float(value)))


def clamp_card(draft: dict) -> Optional[dict]:
    """Snap a model draft onto a valid StrategyCard. None if it cannot be clamped."""
    if not isinstance(draft, dict):
        return None
    structure = str(draft.get("structure") or draft.get("structure_template") or "").strip()
    underlying = str(draft.get("underlying") or "").strip().upper()
    if not underlying:
        sel = draft.get("underlying_selector")
        if isinstance(sel, list) and sel:
            underlying = str(sel[0]).strip().upper()
        elif isinstance(sel, str):
            underlying = sel.strip().upper()
    if structure not in STRUCTURES or not underlying:
        return None
    raw_dte = draft.get("dte", draft.get("dte_bucket", 7))
    raw_delta = draft.get("delta", draft.get("target_delta_bucket", 0.35))
    raw_wing = draft.get("wing_width", draft.get("wing_width_bucket", 10))
    try:
        dte = int(_nearest(float(raw_dte or 7), ALLOWED_DTE))
        delta = float(_nearest(float(raw_delta or 0.35), ALLOWED_DELTA))
        wing = float(_nearest(float(raw_wing or 10), ALLOWED_WING))
        card = StrategyCard.from_dict({
            "underlying": underlying,
            "structure": structure,
            "dte": dte,
            "delta": delta,
            "wing_width": wing if structure == "credit_put_spread" else 10.0,
        })
    except (TypeError, ValueError):
        return None
    out = {
        "underlying": card.underlying,
        "structure": card.structure,
        "dte": card.dte,
        "delta": card.delta,
        "wing_width": card.wing_width if card.structure == "credit_put_spread" else None,
    }
    thesis = str(draft.get("thesis") or "").strip()
    if thesis:
        out["thesis"] = thesis[:400]
    return out


def _tool_list_universe(ctx, **_) -> dict:
    return {"universe": list(ctx.get("universe") or DEFAULT_UNIVERSE)}


def _tool_get_book(ctx, **_) -> dict:
    return {
        "structure_counts": ctx.get("structure_counts") or {},
        "n_book": ctx.get("n_book") or 0,
        "book": list(ctx.get("book") or [])[:40],
    }


def _tool_get_earnings(ctx, **_) -> dict:
    return {
        "earnings_n": ctx.get("earnings_n") or 0,
        "earnings_symbols": (ctx.get("earnings_symbols") or [])[:12],
    }


def _tool_check_novelty(ctx, *, underlying="", structure="", **_) -> dict:
    want_u = str(underlying or "").strip().upper()
    want_s = str(structure or "").strip()
    for row in (ctx.get("book") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("underlying") or "").strip().upper() != want_u:
            continue
        if str(row.get("structure") or "").strip() != want_s:
            continue
        return {"novel": False, "collision": f"{want_s}:{want_u}"}
    return {"novel": True}


_TOOL_REGISTRY = {
    "list_universe": _tool_list_universe,
    "get_book": _tool_get_book,
    "get_earnings": _tool_get_earnings,
    "check_novelty": _tool_check_novelty,
}

_READ_TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": name,
        "description": f"Read-only: {name.replace('_', ' ')}.",
        "parameters": (
            {"type": "object", "properties": {
                "underlying": {"type": "string"},
                "structure": {"type": "string"},
            }, "required": []}
            if name == "check_novelty"
            else {"type": "object", "properties": {}}
        ),
    }}
    for name in _TOOL_REGISTRY
]

_SUBMIT_TOOL = {"type": "function", "function": {
    "name": "submit_cards",
    "description": "Submit the final strategy cards (discrete fields + thesis only; "
                   "NEVER any risk / max-loss / premium number — Python computes those).",
    "parameters": {"type": "object",
                   "properties": {"cards": {"type": "array", "items": _CARD_SCHEMA}},
                   "required": ["cards"]},
}}


def default_ctx(*, universe: Optional[List[str]] = None, book: Optional[List[dict]] = None) -> dict:
    names = [str(s).strip().upper() for s in (universe or DEFAULT_UNIVERSE) if str(s).strip()]
    rows = [r for r in (book or []) if isinstance(r, dict)]
    counts: Dict[str, int] = {}
    for r in rows:
        s = str(r.get("structure") or "").strip()
        if s:
            counts[s] = counts.get(s, 0) + 1
    return {
        "universe": names or list(DEFAULT_UNIVERSE),
        "book": rows,
        "n_book": len(rows),
        "structure_counts": counts,
        "earnings_n": 0,
        "earnings_symbols": [],
    }


def _brief(ctx: dict, n_target: int) -> str:
    uni = ", ".join(ctx.get("universe") or DEFAULT_UNIVERSE)
    lines = [
        f"Propose {n_target} complementary strategy cards that diversify the tournament book.",
        _VENUE,
        f"Universe: {uni}.",
        f"Book: {ctx.get('n_book') or 0} existing card(s). Use get_book and check_novelty before you submit.",
    ]
    sc = ctx.get("structure_counts") or {}
    if sc:
        lines.append("Live structure mix: " + ", ".join(f"{k}={v}" for k, v in sc.items()))
    if ctx.get("earnings_n"):
        syms = ",".join(str(s) for s in (ctx.get("earnings_symbols") or [])[:8] if s)
        lines.append(f"Earnings next 7d: {ctx.get('earnings_n')} names ({syms}).")
    return "\n".join(lines)


def _role_call(client: FeatherlessClient, model: str, system: str, user: str, *,
               max_tokens: int = 500, fallback: Optional[str] = None) -> str:
    for m in ([model] + ([fallback] if fallback and fallback != model else [])):
        try:
            r = client.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                model=m, temperature=0.2, max_tokens=max_tokens,
            )
            if r.content.strip():
                return r.content.strip()
        except LLMError:
            continue
    return ""


def _coerce_cards(obj: Any) -> List[dict]:
    if isinstance(obj, dict):
        if isinstance(obj.get("cards"), list):
            return [g for g in obj["cards"] if isinstance(g, dict)]
        if isinstance(obj.get("genomes"), list):
            return [g for g in obj["genomes"] if isinstance(g, dict)]
        if obj.get("structure") or obj.get("structure_template"):
            return [obj]
    if isinstance(obj, list):
        return [g for g in obj if isinstance(g, dict)]
    return []


def _run_tool_loop(client: FeatherlessClient, model: str, messages: list, ctx: dict,
                   *, temperature: float) -> List[dict]:
    tools = _READ_TOOLS_SPEC + [_SUBMIT_TOOL]
    for _turn in range(MAX_TOOL_TURNS):
        try:
            resp = client.chat(messages, model=model, tools=tools, tool_choice="auto",
                               temperature=temperature, max_tokens=900)
        except LLMError:
            return []
        calls = resp.tool_calls or []
        if not calls:
            return _coerce_cards(resp.json())
        messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": calls})
        submitted: Optional[List[dict]] = None
        for tc in calls:
            fn = (tc.get("function") or {})
            name = str(fn.get("name") or "")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}
            args = args if isinstance(args, dict) else {}
            if name in ("submit_cards", "submit_genomes"):
                submitted = _coerce_cards(args)
                result = {"accepted": len(submitted)}
            else:
                handler = _TOOL_REGISTRY.get(name)
                result = handler(ctx, **args) if handler else {"error": f"unknown tool {name}"}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id") or name,
                "name": name,
                "content": json.dumps(result),
            })
        if submitted is not None:
            return submitted
    return []


def _propose(client: FeatherlessClient, ctx: dict, n_target: int, notes: str) -> List[dict]:
    system = (
        "/no_think You are an options-tournament PROPOSER. Investigate the book with the "
        "read-only tools, then call submit_cards with the final list. Output ONLY discrete "
        "card fields + a short thesis. Do NOT compute or emit any risk, max-loss, premium, "
        "or payoff number — Python owns all risk math."
    )
    user = _brief(ctx, n_target) + (f"\n\nSpecialist notes (hints):\n{notes[:800]}" if notes else "")
    seed = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    drafts = _run_tool_loop(client, PROPOSER_MODEL, [dict(m) for m in seed], ctx, temperature=0.3)
    if drafts:
        return drafts
    try:
        resp = client.chat(
            [dict(m) for m in seed], model=PROPOSER_MODEL, tools=[_SUBMIT_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_cards"}},
            temperature=0.4, max_tokens=1400)
        return _coerce_cards(resp.json())
    except LLMError:
        return []


def _critique_and_revise(client: FeatherlessClient, ctx: dict, drafts: List[dict]) -> List[dict]:
    if not drafts:
        return drafts
    try:
        crit = client.chat(
            [{"role": "system", "content":
              "You are a red-team critic. Return JSON {\"flags\":[{\"index\":i,\"reason\":\"...\"}]} "
              "for cards that are near-duplicates of the book, collide with an earnings window, "
              "or are not submittable at the venue. FLAGS ONLY — never a risk number."},
             {"role": "user", "content":
              "Cards:\n" + json.dumps(drafts)[:6000] + "\n\n" + _brief(ctx, len(drafts))}],
            model=CRITIC_MODEL, temperature=0.0, max_tokens=400).json()
    except LLMError:
        return drafts
    flags = crit.get("flags") if isinstance(crit, dict) else None
    if not isinstance(flags, list) or not flags:
        return drafts
    flagged = {int(f["index"]) for f in flags
               if isinstance(f, dict) and str(f.get("index", "")).lstrip("-").isdigit()}
    if not flagged:
        return drafts
    keep = [d for i, d in enumerate(drafts) if i not in flagged]
    need = len(drafts) - len(keep)
    if need <= 0:
        return drafts
    reasons = "; ".join(str(f.get("reason", ""))[:60] for f in flags if isinstance(f, dict))
    revised = _propose(
        client, ctx, need,
        notes=f"Replace {need} rejected cards (reasons: {reasons}). "
              f"Avoid: {json.dumps([drafts[i] for i in flagged])[:1500]}")
    return (keep + revised) or drafts


def _trace(sink, role: str, model: str, content: str) -> None:
    if sink is not None and content:
        sink.append({"role": role, "model": model, "content": str(content)[:4000]})


def run_specialists(ctx: dict, n_target: int,
                    client: Optional[FeatherlessClient] = None,
                    trace_sink: Optional[List[dict]] = None) -> List[Tuple[dict, str]]:
    """Specialist loop → list of (clamped card, proposer model). Empty on total failure."""
    client = client or FeatherlessClient()
    brief = _brief(ctx, n_target)
    hyp = _role_call(
        client, HYPOTHESIS_MODEL,
        "You are a quant researcher. Given the universe and the current book mix, propose "
        "2-4 complementary underlyings and short directional theses that diversify structure. "
        "Reason briefly, then end with a line 'ANSWER:' followed by terse bullets. No risk numbers.",
        brief, max_tokens=1400, fallback=STRATEGIST_MODEL)
    _trace(trace_sink, "hypothesis", HYPOTHESIS_MODEL, hyp)
    strat = _role_call(
        client, STRATEGIST_MODEL,
        "You are an options strategist. For each thesis, suggest the best structure + delta + "
        "DTE (and wing for spreads) given the book mix and earnings proximity. Terse. No risk numbers.",
        brief + ("\n\nHypotheses:\n" + hyp if hyp else ""))
    _trace(trace_sink, "strategist", STRATEGIST_MODEL, strat)
    notes = "\n".join(x for x in (
        ("Hypotheses:\n" + hyp) if hyp else "",
        ("Strategy:\n" + strat) if strat else "",
    ) if x)
    proposed = _propose(client, ctx, n_target, notes)
    critiqued = _critique_and_revise(client, ctx, proposed)
    if critiqued != proposed:
        _trace(trace_sink, "critic", CRITIC_MODEL,
               f"revised {len(proposed)} → {len(critiqued)} card(s) after novelty/venue review")
    cards: List[Tuple[dict, str]] = []
    for draft in critiqued:
        clamped = clamp_card(draft)
        if clamped:
            cards.append((clamped, PROPOSER_MODEL))
    _trace(trace_sink, "proposer", PROPOSER_MODEL, "\n".join(
        f"- {c.get('structure','?')} {c.get('underlying')} dte={c.get('dte')} "
        f"δ={c.get('delta')} :: {str(c.get('thesis') or '')[:120]}"
        for c, _ in cards))
    return cards

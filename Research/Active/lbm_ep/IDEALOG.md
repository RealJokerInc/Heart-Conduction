# LBM-EP — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
Reopen the LBM cardiac engine work. Foundation (8 phases, 34 tests, D2Q5/D2Q9, BGK/MRT) is in place — now push into engine maturation: anisotropy correctness, boundary artifact modeling, and tau/MRT tuning.

## Next Step
Audit current LBM V1 state — what works, what's brittle, what hasn't been tested. Identify highest-leverage gap (anisotropy vs boundary vs tuning) and start there.

## Thread

### 2026-04-19 — Reopen
- Originally completed 2026-03-16 as `lbm_cardiac`. Closed as "Yes for monodomain, feasible for bidomain."
- Reopened with new framing: not "can it work?" but "is it good enough as a production solver?"
- User scope: anisotropy + boundary artifacts + tuning. Bidomain LBM stays deferred.
- Renamed folder `lbm_cardiac` → `lbm_ep`. KNOWLEDGE.md preserved as Foundation section.

## Failed Approaches
*(none recorded yet for this reopen — see KNOWLEDGE.md foundation for prior decisions)*

## Session Log

### 2026-04-19 — Scaffolding
Reopened question, moved Complete/lbm_cardiac → Active/lbm_ep, updated MASTER.md, preserved literature/papers/KNOWLEDGE.md as foundation. No engine work yet.

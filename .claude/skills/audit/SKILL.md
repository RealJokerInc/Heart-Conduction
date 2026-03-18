---
name: audit
description: Adversarial review via Opus subagent. Reviews any document for completeness, dependency errors, missed edge cases, and domain-specific anti-patterns. Opt-in, never auto-triggered.
argument-hint: "[path to document, or blank for current question's PLAN.md]"
---

# Adversarial Audit

Spawn an Opus subagent to adversarially review a document. The auditor never generated the document — separate context, separate agent, read-only tools.

Input: $ARGUMENTS

---

## Step 1: Determine the Target Document

**If argument is a file path**: Use that path directly.
**If argument names a research question**: Resolve to `Research/Active/{question}/PLAN.md`.
**If no argument**: Default to the active research question's `PLAN.md`. If none active, ask the user.

Verify the target file exists. If not, report the error and stop.

## Step 2: Determine Document Type

Tailor review criteria based on document type:

| Document | Extra checks |
|----------|-------------|
| `PLAN.md` | Each step has all 7 required sections (see PLAN.md addendum below) |
| `KNOWLEDGE.md` | Findings concrete, open questions actionable, no stale entries |
| `IMPLEMENTATION.md` | Phase specs have validation tables, ABC cross-references correct |
| `EXPERIMENT.md` | Backlinks to research question and MASTER.md exist |
| Source code (`.py`) | Float64 consistency, V5.3 not modified, imports resolve |
| Other | General completeness and consistency only |

## Step 3: Spawn Opus Auditor

Use the Agent tool with **model: opus**. The subagent prompt must instruct read-only tool usage and include all review criteria.

**Subagent prompt template** (substitute `{target_path}` and `{plan_md_addendum}`):

```
You are an adversarial auditor. Use ONLY Read, Grep, and Glob tools. DO NOT use Edit, Write, Bash, or any tool that modifies files.

Review: {target_path}

Read the document in full, then check for:

1. COMPLETENESS — Missing steps/phases/sections implied by the structure. Referenced files that don't exist (verify with Glob/Grep). Coverage gaps.

2. DEPENDENCY ERRORS — Wrong ordering (step depends on a later step). Circular dependencies. References to undefined files, functions, or concepts.

3. MISSED EDGE CASES — Unhandled boundary conditions. Unconsidered error paths. Unvalidated assumptions.

4. DOMAIN ANTI-PATTERNS (cardiac electrophysiology simulation)
   - float64: any float32 without explicit justification
   - V5.3 protection: instructions that would modify Monodomain/Engine_V5.3/
   - cardiac_core dedup: duplicate ionic model code instead of shared imports
   - EXPERIMENT.md backlinks: experiments without research question links
   - No verification: implementation steps with no way to confirm success
{plan_md_addendum}
Severity levels:
- CRITICAL: Will cause failure or data loss
- HIGH: Likely problems or significant rework
- MEDIUM: Should fix but won't block progress
- LOW: Style, clarity, minor improvement

Output a severity-sorted list. Under each level, list issues as "- [description] (location)". Write "None found." if empty. End with: "N issues found (X critical, Y high, Z medium, W low)."
```

**PLAN.md addendum** (include only when document type is PLAN.md, otherwise omit):
```
5. PLAN.md STRUCTURE — each step requires these 9 sections:
   Read First, Why, Implementation Spec, Pseudocode, Test Spec, Checklist, Verify, Exit Criteria, Risk.
   Flag any step missing any section.
```

## Step 4: Present Results

Display the subagent's severity-sorted issue list verbatim. Append:

```
Note: This audit is read-only and advisory. Tool restriction is prompt-based
(honor system), not system-enforced. No files were modified.
```

Do not automatically fix any issues. The user decides what to act on.

---

## Rules

- **Never auto-trigger.** Only runs when the user explicitly invokes `/audit`.
- **Read-only.** Auditor must not modify files — enforced by prompt instruction (honor system).
- **Separate context.** Auditor has no prior conversation knowledge — reviews the document cold.
- **No author bias.** Auditor never generated the document. Approaches it as a skeptical outsider.
- **Present all findings.** Do not filter or editorialize the auditor's output.

# Project Rules

## 1. Direction and scope

- Stay close to `__docs__/paper.md` and `__docs__/plan.md` and the approved research direction.
- Do not invent results, metadata, citations, commands, or explanations.
- Do not change the method, dataset role, experiment scope, or evaluation protocol because an inspection or experiment is inconvenient.
- If an unexpected failure, contradiction, missing asset, permission issue, or dataset inconsistency appears, report it to the user immediately and wait for direction when it could affect the study.
- There is no fallback or backup research plan. If the main method fails, diagnose and fix that method. Do not silently substitute another method.

## 2. Command ownership

Every command must be classified before execution:

### Agent-typed commands (smoke checks)

The agent may run only lightweight, read-only checks needed to inspect files, validate assumptions, check syntax, inspect metadata, or test a small representative example. Examples include listing files, reading headers, counting files, checking a few labels, and running a narrow syntax or unit check.

### User-typed commands (full runs)

The user must run full training, full preprocessing, dataset-wide conversion, multi-seed experiments, active-learning trajectories, long profiling jobs, downloads, and any command that consumes substantial compute, memory, time, or storage. The agent must not execute these commands.

The agent may prepare or display a user-typed command, but must label it clearly and must not run it on the user's behalf.

## 3. Evidence and failure handling

- Separate observed facts, documented facts, assumptions, and unresolved checks.
- Validate file formats and representative metadata before designing preprocessing around them.
- Never infer a missing image, label, split, or metric from a filename or README alone.
- If a smoke check crashes, times out, produces contradictory output, or reveals a missing dependency, stop that line of investigation and report the exact failure. Do not hide it by trying unrelated approaches or changing the research direction.
- A successful smoke check does not replace a full experiment or user-run validation.

## 4. Stage protocol and reporting

Every stage must end with a short report containing:

1. **Status:** complete, blocked, or failed.
2. **What was checked or changed.**
3. **Observed result and unresolved issues.**
4. **Agent-typed commands:** the exact lightweight commands actually run.
5. **User-typed commands:** exact commands prepared for the user, or `None` if none were prepared. These must never be described as executed by the agent.
6. **Next step:** only the next step already supported by `paper.md` or explicitly approved by the user.

If a stage cannot be completed, end the report as **blocked** or **failed** and state the concrete reason. Do not mark it complete by replacing the main method with an unapproved alternative.

## 5. Documentation and memory

- Keep documentation precise, concise, and tied to observed evidence.
- Keep memory short: store only durable project facts, constraints, or verified commands needed for future work.
- Do not store speculative interpretations, long logs, or unverified results in memory.
- no fallback or backup plan of anykind, if the main method fail, we fix
- each command has two types: agent-typed (smoke) and user typed (full run, etc). agent must never execute user typed command 
- every stage ends with complete with a report that specify the commands into two mentioned types
- stay close to the paper.md, if anything change or unexpected failure occur, report to the user immediately and do not change the direction your own. Do not hallucinate
- keep the memory short and brief, do not write too much into the memory to avoid hallucination
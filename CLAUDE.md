# Claude Code Guidelines for This Project

## Critical Lessons Learned (2 days of debugging)

### 1. ALWAYS DELEGATE NON-TRIVIAL CHANGES TO SUBAGENTS
**Why:** Direct implementation leads to cascading failures that only become visible after hours of wasted training time.

**What this means:**
- Changes that affect core training loop → delegate with Agent(subagent_type=general-purpose)
- Changes that add new failure modes → delegate for investigation first
- Changes affecting multiple files → delegate for impact analysis
- Observability/monitoring additions → especially delegate to catch unintended side effects

**What NOT to do:**
- Don't add "safety checks" that create deadlocks (e.g., startup validation in training loop)
- Don't add logging that changes code paths
- Don't refactor error handling without testing

### 2. DOCUMENT ROOT CAUSES IN MEMORY, NOT CODE
When a fix is implemented, save **why it failed** to `memory/` directory:
- Root cause analysis
- What didn't work and why
- What worked and why
- Performance implications

**Files created:**
- `memory/streaming_training_fixes.md` — np.savez_compressed incomplete writes on Windows

### 3. FILE I/O ON WINDOWS IS FRAGILE
**Symptoms:** "File not found after write", "BadZipFile: incomplete", file corruption under load
**Root cause:** np.savez_compressed returns before flushed to OS disk cache
**Solution that works:** tempfile + os.fsync() + atomic rename
**Solution that doesn't work:**
- Simple wait loops (arbitrary timing)
- File size stability checks alone (insufficient)
- Higher wait times (just slow, not reliable)

### 4. AVOID ADDING SAFETY CHECKS TO TRAINING LOOPS
Two attempted "fixes" that made things worse:
- Startup validation: Created deadlock when queue filled during validation
- Observability logging in train_epoch: Broke code path, hung immediately

**Why:** Training loop is already fragile - adding ANY blocking calls creates new failure modes.

**Better approach:** Delegate to separate monitoring processes, don't modify core loop.

### 5. CONTINUOUS MONITORING MUST BE EXTERNAL
**Don't:** Add checks inside training loop
**Do:** Use external Monitor (tail -f log | grep) with notifications every 30s

This session: Set up continuous monitor that checks:
- Log file line count (training progress)
- Process alive (crashes detected immediately)
- Epoch completion lines

### 6. BASH vs POWERSHELL COMMAND DIFFERENCES
**Critical:** When running bash commands via Bash tool, use bash syntax, NOT PowerShell syntax.

**Common mistakes:**
- ❌ `Tee-Object` (PowerShell) → ✅ `tee` (bash)
- ❌ `Get-Content file -Tail 20` (PowerShell) → ✅ `tail -20 file` (bash)
- ❌ `Select-String` (PowerShell) → ✅ `grep` (bash)
- ❌ `Get-Process` (PowerShell) → ✅ `ps aux` (bash)

**Rule:** 
- **Bash tool:** Use bash/POSIX syntax (grep, tail, ps, tee)
- **PowerShell tool:** Use PowerShell cmdlets (Get-Content, Select-String, Get-Process)
- **Don't mix:** Mixing causes "command not found" errors and silent failures

**Example fix:**
```bash
# ❌ WRONG: Bash tool with PowerShell syntax
python script.py 2>&1 | Tee-Object -FilePath output.log

# ✅ CORRECT: Bash tool with bash syntax  
python script.py 2>&1 | tee -a output.log
```

## Current Status

**Streaming Training System (WORKING — not "hanging", just slow on CPU):**
- 4 parallel workers generate unlimited batches; trainer is the bottleneck.
- fsync-based atomic writes (0.5s stability wait).
- File-based queue, capped at 16 files advisory (workers warn but don't block — non-blocking back-pressure).
- Queue dir is per-run (`data/gen_queue_<timestamp>/`) and now auto-removed on shutdown; startup also reaps anything older than 1 hour.
- **Realistic per-step time: ~140s on this CPU** (after `src/ml/model.py` forward-pass vectorization landed 2026-05-15). Earlier "1–1.5 hours for 30 epochs" estimate was wrong; with 50 steps/epoch × 30 epochs × 140s that's ~60 hours on CPU. For real runs, move to GPU or shrink `window_size`/sonar resolution.
- The "stall every 32 samples" the previous session chased was not a leak — it's the natural cycle of one training step (4 batch files of 8 samples + one forward+backward+opt). See `memory/previous_agent_misdiagnosis_corrected.md`.

**To Avoid Future Issues:**
1. Don't modify train_streaming.py or train.py without subagent review.
2. Add monitoring only via external processes.
3. When perf looks "degraded", plot the metric over time first — if it cycles cleanly, name the cycle, don't call it a leak.
4. Profile compute cost (forward / backward / step) before reaching for `gc.collect()` or memory-leak hypotheses.
5. Test fixes on minimal examples first (`--workers 2 --epochs 1 --steps 2`).

## Deployment Checklist

Before running long training:
- [ ] Run minimal test (2 workers, 1 epoch, 2 steps)
- [ ] Verify first epoch completes (not hangs at startup)
- [ ] Start continuous monitor (Monitor tool, 30s intervals)
- [ ] Let training run without manual intervention
- [ ] Any changes → delegate to Agent first

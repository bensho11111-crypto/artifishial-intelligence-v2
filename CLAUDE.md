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

## Current Status

**Streaming Training System (WORKING):**
- 4 parallel workers generate unlimited batches
- fsync-based atomic writes (0.5s stability wait)
- File-based queue (~1 GB max disk)
- Expected: 1-1.5 hours for 30 epochs

**To Avoid Future Issues:**
1. Don't modify train_streaming.py or train.py without subagent review
2. Add monitoring only via external processes
3. When bugs occur, investigate root cause first (delegate to Agent)
4. Document findings in memory before implementing fix
5. Test fixes on minimal examples first (--workers 2 --epochs 1 --steps 2)

## Deployment Checklist

Before running long training:
- [ ] Run minimal test (2 workers, 1 epoch, 2 steps)
- [ ] Verify first epoch completes (not hangs at startup)
- [ ] Start continuous monitor (Monitor tool, 30s intervals)
- [ ] Let training run without manual intervention
- [ ] Any changes → delegate to Agent first

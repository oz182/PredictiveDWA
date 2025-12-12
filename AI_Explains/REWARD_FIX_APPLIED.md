# Reward Function Fix - Applied

## ✅ Simple Fix Implemented

### What Was Changed

In `src/learning/train.py`, the `compute_reward()` function now properly handles all overlap cases:

**Before (BUG):**
```python
if overlap_type == 'none':
    reward = -0.1
elif overlap_type == 'person':
    reward = -1.0
elif overlap_type == 'both':
    reward = -1.0  # Same penalty as person alone!
else:
    reward = -0.1  # Door overlap fell here - no penalty!
```

**After (FIXED):**
```python
if overlap_type == 'none':
    reward = -0.1
elif overlap_type == 'person':
    reward = -1.0
elif overlap_type == 'door':
    reward = -1.0  # ✓ Now explicitly penalized
elif overlap_type == 'both':
    reward = -2.0  # ✓ Worse than either alone
else:
    reward = -0.1
```

### What This Fixes

1. **Door overlaps are now penalized** (`-1.0` instead of `-0.1`)
2. **Both overlaps are penalized more** (`-2.0` instead of `-1.0`)
3. **Agent will learn to avoid doors** across all curriculum stages

### Impact on Training

**Before Fix:**
- ❌ Agent had no incentive to avoid door inflation zones
- ❌ Door collision rate would be high
- ❌ Curriculum learning ineffective for door avoidance

**After Fix:**
- ✅ Agent penalized for entering door zones
- ✅ Strong learning signal for door avoidance
- ✅ Curriculum will teach radius-aware avoidance

### Expected Behavior

During training, you should now see:
- **`overlap_door_pct`** decreasing over episodes
- Agent maintaining clearance from doors
- Performance improving across all door radii (small to large)

### Next Steps

1. **Start training** with the fixed reward function:
   ```bash
   cd src/learning
   python train.py --episodes 500 --use-wandb
   ```

2. **Monitor these metrics** in WandB:
   - `overlap_door_pct`: Should trend downward
   - `overlap_free_pct`: Should trend upward
   - `return`: Should improve as agent learns avoidance
   - `door_halo_radius`: Should show increasing variance with curriculum

3. **If door collisions remain high** (>15% after 100 episodes):
   - Consider implementing radius-aware clearance reward (Option B)
   - See `REWARD_FUNCTION_UPDATE.md` for advanced implementations

### Optional Enhancements

If the simple fix isn't sufficient, you can add:

**Radius-Aware Clearance Gradient** (provides smoother learning signal):
- Rewards maintaining proportional clearance
- Larger doors automatically require more clearance
- See full implementation in `REWARD_FUNCTION_UPDATE.md`, Option B

### Validation

- ✅ Syntax check: Passed
- ✅ Linter check: No errors
- ✅ All overlap types handled: Yes
- ✅ Backward compatible: Yes

---

## Summary

The critical bug where **door overlaps weren't penalized** has been fixed. The agent will now:
1. Learn to avoid door inflation zones
2. Distinguish between single and double overlaps
3. Benefit from curriculum learning across different door radii

This simple fix should be sufficient for most cases. If you need more sophisticated radius-aware behavior, the advanced options are documented and ready to implement.

**You're now ready to train with proper door avoidance!** 🚀




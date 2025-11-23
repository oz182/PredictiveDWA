# Door Training Changes - Summary

## Overview
Modified training files to remove door overlap penalties from rewards while keeping door inflation radius in costmap and planner. The goal is to test if the agent can learn to avoid doors implicitly by avoiding people who emerge from doors.

## Changes Made

### 1. Feature Extraction (All 3 Training Files)

**Added: Door Distance Feature**
- Euclidean distance from robot to door
- Added to state representation for agent awareness
- Normalized appropriately for each training file

**Files Modified:**
- `src/learning/train.py` - Line 106: Added `door_distance`
- `src/learning/train_theta_range.py` - Line 82: Added `door_distance`
- `src/learning/train_theta_range_v1.py` - Line 46: Added `door_distance` (normalized 0-1)

**Input Dimension Change:**
- **train.py**: Increased by 1 feature (door_distance)
- **train_theta_range.py**: Increased by 1 feature (door_distance)
- **train_theta_range_v1.py**: Increased by 1 feature (door_distance)

### 2. Overlap Detection

**Modified: `check_robot_overlap()` Function**
- Removed all door overlap detection logic
- Now only checks person overlaps
- Returns simplified dict with only 'person_overlap' and 'overlap_type'
- `overlap_type` now only has values: 'none' or 'person'

**Files Modified:**
- `src/learning/train.py` - Lines 115-190
- `src/learning/train_theta_range.py` - Lines 91-166
- `src/learning/train_theta_range_v1.py` - Lines 74-149

### 3. Reward Function

**Modified: `compute_reward()` Function**

**Removed:**
- All door-specific penalties
- Door overlap penalties (-0.3, -1.0, -0.8, depending on file)
- 'both' overlap type handling

**Added:**
- `door_distance` to info dict for logging (not penalized)
- Comments explaining door is not penalized

**Kept:**
- Person overlap penalties (unchanged)
- Progress rewards (unchanged)
- Goal bonuses (unchanged)
- Collision penalties (unchanged)
- In `train_theta_range_v1.py`: `door_penetration` still computed for logging

**Files Modified:**
- `src/learning/train.py` - Lines 193-264
- `src/learning/train_theta_range.py` - Lines 169-227
- `src/learning/train_theta_range_v1.py` - Lines 284-347

### 4. Training Loop

**Modified:**
- Changed overlap_counts from `{'none': 0, 'person': 0, 'door': 0, 'both': 0}`
- To: `{'none': 0, 'person': 0}`
- Removed door overlap statistics from print statements
- Removed door overlap statistics from wandb logging

**Files Modified:**
- `src/learning/train.py` - Lines 356, 430-432, 435-444
- `src/learning/train_theta_range.py` - Lines 353, 397-398, 400-409

## What Stays Unchanged

### Door Infrastructure (Kept Intact)
1. **Global Planner (`src/algo/global_planner.py`)**: 
   - `door_halo_radius` parameter still used
   - Path planning still avoids door
   - Door influence on path generation unchanged

2. **Robot Costmap (`src/sim/robot.py`)**:
   - Door semicircular inflation still added to costmap (lines 259-320)
   - DWA local planner still sees door as obstacle
   - Door appears in robot's egocentric perception

3. **Navigation Features**:
   - Door position (x, y) in robot frame still included
   - Door angle still included
   - Agent still "knows" where door is

## Expected Behavior

### What Should Happen
1. **Agent perceives door**: Door visible in costmap and as feature
2. **DWA avoids door locally**: Local planner still treats door as obstacle
3. **No explicit door penalty**: Agent not explicitly punished for door proximity
4. **Implicit learning**: Agent should learn door = people zone = avoid

### What to Monitor
1. **Door penetration rate**: Track even without penalty (in info dict)
2. **Person penetration rate**: Should stay low
3. **Learning convergence**: May need more episodes
4. **Emergent behavior**: Does agent still avoid door?

## Training Implications

### Input Dimension Change
**CRITICAL**: All networks need to accept new input dimension (+1 feature)
- Training from scratch: Set correct input_dim in network initialization
- Loading old checkpoints: Will fail due to dimension mismatch

### Hyperparameter Considerations
- May need increased person penalty to compensate
- May need more training episodes for implicit learning
- May need higher people spawn rate near door

### Experiment Variations
1. **Baseline**: Train with current changes
2. **High spawn rate**: Increase people spawn near door
3. **Stronger person penalty**: Increase person penalty magnitude
4. **Longer training**: Increase episode count

## Files Summary

### Modified Files (3)
✓ `src/learning/train.py`
✓ `src/learning/train_theta_range.py`
✓ `src/learning/train_theta_range_v1.py`

### Unchanged Files
○ `src/sim/robot.py` - Door costmap still active
○ `src/algo/global_planner.py` - Door avoidance still active
○ `src/sim/sim.py` - Simulation setup unchanged
○ `src/sim/person.py` - Person behavior unchanged

## Key Differences Between Training Files

### train.py (PPO with offset)
- Action: Continuous offset [-π/3, π/3]
- Feature count: 17 → 18 (added door_distance)
- Removed door penalty: -1.0
- Still logs: door_distance

### train_theta_range.py (DQN with discrete theta)
- Action: Discrete theta_range [10°, 20°, 30°, 45°]
- Feature count: 13 → 14 (added door_distance)
- Removed door penalty: -0.3
- Still logs: door_distance

### train_theta_range_v1.py (DQN with continuous penetration)
- Action: Discrete theta_range [10°, 20°, 30°, 45°]
- Feature count: 10 → 11 (added normalized door_distance)
- Removed door penalty: -0.5 * (penetration^1.5)
- Still logs: door_penetration (for analysis)
- Still computes: `compute_door_penetration()` (logging only)

## Testing the Changes

### Before Training
```bash
# Verify door is still in costmap (should see door visualization)
python run_gui.py

# Check input dimensions match network architecture
python src/learning/train.py --episodes 1 --max-steps 10
```

### During Training
- Monitor door_distance in logs
- Check if door penetration decreases over time
- Compare person vs door penetration rates

### After Training
```bash
# Evaluate trained policy
python src/learning/test_theta_eval.py

# Compare metrics: with vs without door penalties
```

## Scientific Hypothesis

**Hypothesis**: If people consistently emerge from doors, the agent will learn that door zones correlate with high person density and will avoid doors implicitly through person-avoidance behavior, without explicit door penalties.

**Success Criteria**:
1. Door penetration rate remains low (similar to baseline)
2. Person penetration rate remains low
3. Agent reaches goal efficiently
4. Generalizes to varying people densities

## Notes

- All changes preserve existing infrastructure
- Door is still physical obstacle in costmap
- Only reward signal changed
- Can revert by re-adding door penalties
- Input dimension increased by 1 in all files


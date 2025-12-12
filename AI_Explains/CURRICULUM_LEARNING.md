# Curriculum Learning Implementation

## Overview

This implementation adds **curriculum learning** with **domain randomization** to make the agent generalize across different door configurations instead of memorizing a single static environment.

## What Changed

### 1. **Robot Initialization** (`src/sim/robot.py`)

**Before:**
```python
def __init__(self, position, radius, corridor_bounds, door_position):
    # ...
    door_halo_radius=1.8,  # Hardcoded
```

**After:**
```python
def __init__(self, position, radius, corridor_bounds, door_position, door_halo_radius=1.8):
    # ...
    door_halo_radius=door_halo_radius,  # Parameterized
```

**Impact:** Robot can now be initialized with different door inflation radiuses.

---

### 2. **Simulation Initialization** (`src/sim/sim.py`)

**Before:**
```python
def __init__(self, corridor_width=5.0, door_side="right", num_people=5, people_speeds=None):
    self.door_position = 0.4 * self.corridor_length  # Fixed at 40%
    self.door_side = door_side  # Must be specified
```

**After:**
```python
def __init__(self, corridor_width=5.0, door_side=None, num_people=5, people_speeds=None,
             door_halo_radius=None, door_position_x=None):
    # Randomize door position if not specified (25%-65% of corridor)
    if door_position_x is None:
        self.door_position = random.uniform(0.25 * self.corridor_length, 0.65 * self.corridor_length)
    
    # Randomize door side if not specified
    if door_side is None:
        self.door_side = random.choice(["left", "right"])
    
    # Randomize door radius if not specified (0.8m - 2.5m)
    if door_halo_radius is None:
        self.door_halo_radius = random.uniform(0.8, 2.5)
```

**Impact:** 
- Door can be positioned anywhere along the corridor
- Door can be on left or right side
- Door inflation radius varies each episode
- Persons automatically spawn from the randomized door position

---

### 3. **Curriculum Scheduler** (`src/learning/train.py`)

New function that returns episode-specific parameters:

```python
def get_curriculum_params(episode: int, total_episodes: int) -> dict
```

**Curriculum Stages:**

| Stage | Episodes (%) | Radius Range (m) | Position Range (m) | Side | Difficulty |
|-------|-------------|------------------|-------------------|------|------------|
| **Easy** | 0-20% | 1.6 - 2.0 | 7.0 - 9.0 | Right only | ⭐ |
| **Medium** | 20-50% | 1.2 - 2.3 | 6.0 - 11.0 | Right only | ⭐⭐ |
| **Hard** | 50-80% | 0.8 - 2.5 | 5.0 - 13.0 | Left or Right | ⭐⭐⭐ |
| **Expert** | 80-100% | 0.8 - 2.5 | 5.0 - 13.0 | Left or Right | ⭐⭐⭐⭐ |

**Rationale:**
- **Early training (Easy/Medium):** Narrow ranges help agent learn core behaviors
- **Mid training (Hard):** Introduce full diversity to force generalization
- **Late training (Expert):** Continue diverse training to polish policy

---

### 4. **Training Loop Update** (`src/learning/train.py`)

**Before:**
```python
for ep in range(episodes):
    sim = Simulation(
        corridor_width=corridor_width,
        door_side=door_side,  # Fixed
        num_people=num_people,
        people_speeds=[...],
    )
```

**After:**
```python
for ep in range(episodes):
    # Get curriculum parameters
    curriculum = get_curriculum_params(ep, episodes)
    
    # Sample from curriculum ranges
    door_halo_radius = random.uniform(*curriculum['door_halo_radius_range'])
    door_position_x = random.uniform(*curriculum['door_position_x_range'])
    door_side_param = None if curriculum['randomize_door_side'] else 'right'
    
    sim = Simulation(
        corridor_width=corridor_width,
        door_side=door_side_param,
        num_people=num_people,
        people_speeds=[...],
        door_halo_radius=door_halo_radius,
        door_position_x=door_position_x,
    )
```

---

### 5. **Enhanced Logging**

Training now logs curriculum parameters each episode:

**Console Output:**
```
Episode 125/500 | Return: 245.32 | Steps: 1250
  Curriculum Stage: MEDIUM | Door Radius: 1.85m | Door Pos: 8.3m | Side: right
  Overlaps - Free: 78.2% | Person: 12.4% | Door: 8.1% | Both: 1.3%
  Offsets - Avg: 8.3° | Max: 22.1° | (range: ±30°)
```

**WandB Tracking:**
- `curriculum_stage`: Current difficulty stage
- `door_halo_radius`: Actual door radius used
- `door_position_x`: Actual door position
- `door_side`: Which side door is on

---

## Why This Improves the Agent

### Problem with Original Implementation

1. **Static environment**: Door always at 8.0m with 1.8m radius on right side
2. **Observation ignored**: Agent received radius in observations but it was always 0.6 (normalized)
3. **Memorization**: Agent could memorize the single fixed layout instead of learning the causal relationship
4. **No generalization**: Would fail if deployed with different door configurations

### Solution: Curriculum + Randomization

1. **Forced attention**: Radius varies, so agent MUST use that observation feature
2. **Causal learning**: Agent learns "larger radius = avoid earlier/wider"
3. **Position invariance**: Door position varies, so agent can't rely on fixed spatial memorization
4. **Side invariance**: Door on left or right forces symmetric avoidance strategies
5. **Gradual difficulty**: Curriculum prevents overwhelming the agent early in training

---

## Expected Agent Behaviors After Training

The trained agent should demonstrate **radius-aware adaptive behaviors**:

### Small Radius (0.8 - 1.2m)
- Pass close to door
- Minimal path deviation
- Late avoidance initiation
- Quick return to global path

### Medium Radius (1.5 - 1.8m)
- Moderate clearance
- Balanced deviation
- Standard avoidance timing
- Current learned behavior

### Large Radius (2.0 - 2.5m)
- Wide berth around door
- Early avoidance initiation
- Potentially cross to opposite corridor side
- Gradual return to path

### Position Variance
- Avoid door regardless of x-position
- Consistent clearance maintenance
- No fixed spatial triggers

### Side Invariance
- Symmetric behavior for left/right doors
- Mirror image avoidance patterns
- Equal performance on both sides

---

## Training with Curriculum

### Start Training

```bash
cd src/learning
python train.py --episodes 500 --use-wandb
```

The curriculum will automatically adjust difficulty throughout the 500 episodes.

### Monitor Progress

Watch these metrics in WandB:
- **`door_halo_radius`**: Should show increasing variance over time
- **`door_position_x`**: Should spread from narrow to wide range
- **`curriculum_stage`**: Tracks progression (easy → medium → hard → expert)
- **`overlap_door_pct`**: Should remain low across all stages
- **`return`**: May dip during stage transitions, should recover

---

## Evaluation Strategy

After training, evaluate generalization:

### 1. **Interpolation Test** (Within Training Range)
```python
test_radii = [0.9, 1.5, 2.2]  # Within [0.8, 2.5]
```
**Expected**: High success rate, low door collisions

### 2. **Extrapolation Test** (Outside Training Range)
```python
test_radii = [0.5, 3.0]  # Outside training range
```
**Expected**: Degraded but reasonable performance

### 3. **Fixed Configuration Test** (Consistency)
```python
# Run 20 episodes with identical configuration
for _ in range(20):
    sim = Simulation(..., door_halo_radius=1.5, door_position_x=8.0, door_side='right')
```
**Expected**: Consistent behavior, low variance

### 4. **Ablation Study** (Importance of Radius Feature)
- Remove radius from observation vector
- Retrain and compare performance
**Expected**: Performance should drop significantly, proving the feature is used

---

## Customizing the Curriculum

### Adjust Stage Boundaries

Modify `get_curriculum_params()` in `train.py`:

```python
# Make early training longer
if progress < 0.3:  # Changed from 0.2
    return {..., 'stage': 'easy'}
```

### Adjust Parameter Ranges

```python
# Expand radius range
'door_halo_radius_range': (0.5, 3.0),  # Instead of (0.8, 2.5)

# Narrow position range
'door_position_x_range': (7.0, 10.0),  # Instead of (5.0, 13.0)
```

### Disable Curriculum (Use Full Randomization)

In the training loop:

```python
# Skip curriculum scheduler
door_halo_radius = random.uniform(0.8, 2.5)  # Full range always
door_position_x = random.uniform(5.0, 13.0)  # Full range always
door_side_param = None  # Always randomize
```

---

## Troubleshooting

### Agent performs poorly in hard stages
**Solution**: Extend easy/medium stages or narrow hard stage ranges initially

### High door collision rate
**Solution**: Increase door avoidance penalty in reward function

### Agent ignores radius observation
**Solution**: Add explicit reward shaping based on radius-clearance relationship:
```python
expected_clearance = door_halo_radius + safety_margin
actual_clearance = distance_to_door
clearance_reward = -abs(actual_clearance - expected_clearance)
```

### Performance drops during stage transitions
**Solution**: This is normal! Agent is adapting to increased difficulty. Monitor if it recovers.

---

## Code Files Modified

1. **`src/sim/robot.py`**
   - Added `door_halo_radius` parameter to `__init__()`

2. **`src/sim/sim.py`**
   - Added `door_halo_radius`, `door_position_x`, `door_side` parameters
   - Added randomization logic when parameters are `None`

3. **`src/learning/train.py`**
   - Added `get_curriculum_params()` function
   - Modified training loop to use curriculum
   - Enhanced logging for curriculum parameters

---

## Next Steps

1. **Train with curriculum**: Run 500+ episodes to see adaptation
2. **Evaluate generalization**: Test on unseen configurations
3. **Analyze feature importance**: Verify agent uses radius observation
4. **Fine-tune curriculum**: Adjust stages based on training curves
5. **Consider additional randomization**: Corridor width, person speeds, etc.

---

## Summary

This implementation transforms the training from **memorizing a single environment** to **learning generalizable avoidance behaviors** by:

1. ✅ Varying door inflation radius (0.8m - 2.5m)
2. ✅ Varying door position along corridor (25% - 65%)
3. ✅ Varying door side (left/right)
4. ✅ Using curriculum learning for gradual difficulty increase
5. ✅ Forcing agent to use radius observation feature
6. ✅ Persons automatically spawn from randomized door location

The agent will learn: **"Read the radius → compute required clearance → adjust trajectory accordingly"**

Instead of: **"Always avoid the fixed obstacle at x=8.0m"**




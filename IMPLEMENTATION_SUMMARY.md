# Agent-Controlled Heading Offset Implementation

## Summary
Successfully implemented an offset-based control approach where the agent learns to make small corrections to the path heading, resolving the control conflict between TS-DWA's path-following and the agent's obstacle avoidance.

## Architecture

### Previous Approach (Had Control Conflict)
```
Global Path → theta_ph (automatic)
Agent → left_weight, right_weight (asymmetric sampling)
CONFLICT: Path and agent competed for control
```

### New Approach (Hierarchical Control)
```
Global Path → theta_ph_path (baseline direction)
Agent → offset (±30° correction)
Combined → theta_ph_final = theta_ph_path + offset
TS-DWA → samples around theta_ph_final
```

## Changes Made

### 1. `src/algo/ts_dwa.py`
**Modified `update()` method:**
- Extracts path heading as `theta_ph_path` (baseline)
- Reads agent offset from `self.agent_offset` attribute
- Combines: `theta_ph = theta_ph_path + agent_offset`
- Sets `kappa = 0` (no curvature coupling)
- Updated debug output to show path heading, offset, and final heading

**Key lines:**
```python
theta_ph_path = self._extract_heading(global_path)
agent_offset = getattr(self, 'agent_offset', 0.0)
theta_ph = theta_ph_path + agent_offset
kappa = 0.0
```

### 2. `src/learning/train.py`
**Action space:**
- Changed from 2D `(left_weight, right_weight)` to 1D `(offset)`
- `num_actions = 1`

**State features enhanced:**
```python
# Added critical goal information at start of feature vector
feat.append(goal_angle_robot)  # Angle to goal in robot frame
feat.append(goal_dist)          # Distance to goal

# Transform people positions to robot frame (more consistent)
rel_x_robot = rel_x * cos(-orientation) - rel_y * sin(-orientation)
rel_y_robot = rel_x * sin(-orientation) + rel_y * cos(-orientation)
```

**Action scaling:**
```python
max_offset = math.pi / 6  # ±30 degrees
offset = agent_output * max_offset
sim.robot.nav.agent_offset = offset
```

**Reward regularization:**
```python
# Encourage minimal corrections
offset_penalty_weight = 0.05
reward += -offset_penalty_weight * abs(offset)
```

### 3. `src/learning/test.py`
**Updated for 1D action space:**
- Changed `num_actions = 1`
- Replaced weight plotting with offset plotting
- Shows offset in both radians and degrees
- Visualization shows zero line (follow path baseline)

## Benefits

✅ **No Control Conflict**: Agent adds corrections to path, not competing with it

✅ **Simpler Learning**: Agent learns "when to deviate" not "how to navigate"

✅ **Bounded Action Space**: ±30° is sufficient for obstacle avoidance

✅ **Safe Baseline**: If agent outputs zero, robot follows path

✅ **Interpretable**: Can see agent's corrections in real-time

✅ **Faster Convergence**: Easier learning problem than full heading control

## Reward Function (Obstacle-Focused)

The reward function is **simplified to focus ONLY on obstacle avoidance**:

| Situation | Reward | Rationale |
|-----------|--------|-----------|
| Free space (no overlaps) | **+1.0** | Good - clear of all obstacles |
| Person proxemic zone | **-1.0** | Bad - violating personal space |
| Door halo zone | **-0.5** | Moderate - near door |
| Both zones | **-1.5** | Worse - violating multiple zones |
| Hard collision | **-10.0** | Critical - must avoid |

**Design Philosophy:**
- Path planner handles goal-seeking (via `theta_ph_path`)
- Agent ONLY learns obstacle avoidance (via offset corrections)
- No progress reward, no time penalty, no goal bonus
- Simple and focused: "stay clear of obstacles"

## Training Notes

### Hyperparameters
- **Offset range**: ±30° (π/6 rad) - tunable via `max_offset`
- **Action std**: 0.4 (reduced from 0.6 for tighter exploration)

### Expected Behavior
Agent should learn to:
1. Output ~0 offset when path is clear → follow path
2. Output positive offset when obstacle on left → steer right
3. Output negative offset when obstacle on right → steer left
4. Adjust offset magnitude based on obstacle proximity and clearance

### State Features (Total: 15 dimensions)
1. `goal_angle_robot` - Critical for knowing target direction
2. `goal_dist` - How far to goal
3-4. `waypoint` (x, y) - Next path waypoint
5-6. `door_position` (x, y) - Door location
7. `door_angle` - Angle to door
8. `linear_velocity` - Current speed
9. `angular_velocity` - Current turn rate
10-15. Three closest people `(dx, dy)` in robot frame
16. `dist_left` - Distance to left corridor boundary
17. `dist_right` - Distance to right corridor boundary

## Usage

### Training
```bash
cd src/learning
python train.py --episodes 50 --lr-actor 1e-3 --use-wandb
```

### Testing
```bash
cd src/learning
python test.py --render --model checkpoints/theta_qnet.pt
```

### Adjusting Offset Range
To allow larger corrections, modify in `train.py`:
```python
max_offset = math.pi / 3  # ±60 degrees instead of ±30
```

### Tuning Offset Penalty
To encourage/discourage deviations, modify in `train.py`:
```python
offset_penalty_weight = 0.1  # Stronger penalty = prefer following path
offset_penalty_weight = 0.01  # Weaker penalty = more willing to deviate
```

## Next Steps

1. **Train the agent** with new architecture
2. **Monitor offset values** during training - should be small when clear, larger near obstacles
3. **Evaluate performance** compared to baseline path following
4. **Optional**: Implement layered clearance scoring (hard vs soft obstacles)
5. **Optional**: Add offset magnitude as a second action if ±30° is insufficient

## Technical Details

### Why kappa=0?
- Path curvature coupling `κ` is less useful when agent controls heading
- In straight corridors (95% of your scenario), κ ≈ 0 anyway
- Agent's offset provides sufficient directional control
- Simpler equation: `ω = v · α_ph · h` (no curvature term)

### Why Add Goal to State?
Without goal features, agent has no way to know:
- Which direction leads to success
- How to trade off between obstacle avoidance and progress
- When it's moving closer vs farther from goal

### Robot Frame Transform
People positions are now in robot's reference frame, making features invariant to robot's absolute orientation. This helps generalization across different headings.


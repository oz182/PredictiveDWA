# Reward Function Update for Curriculum Learning

## Problem Analysis

The current reward function has issues that prevent effective curriculum learning:

### Issue 1: Missing Door Overlap Case
```python
if overlap_type == 'none':
    reward = -0.1
elif overlap_type == 'person':
    reward = -1.0
elif overlap_type == 'both':
    reward = -1.0
else:
    reward = -0.1  # Door overlap falls here! Same as free space!
```

**Bug:** Door overlap (`overlap_type == 'door'`) is not explicitly handled, so it defaults to `-0.1` (same as free space).

### Issue 2: Fixed Penalties for Variable Difficulty
- Small radius (0.8m): Easy to avoid, penalty = -1.0
- Large radius (2.5m): Hard to avoid, penalty = -1.0
- **No differentiation!** Agent doesn't learn that larger radii require different strategy.

### Issue 3: No Clearance Gradient
Current reward is binary:
- Inside inflation zone: -1.0
- Outside inflation zone: -0.1

Better reward should provide gradient:
- Far from door: 0.0 (good)
- Moderate distance: -0.2 (acceptable but not optimal)
- Near boundary: -0.5 (risky)
- Inside zone: -1.0 (bad)

---

## Proposed Solution: Radius-Aware Reward Shaping

### Option A: Simple Fix (Minimal Changes)

**Just add the missing door case:**

```python
def compute_reward(sim, progress_prev_dist: float, offset: float = 0.0) -> tuple[float, float, dict]:
    """Reward with explicit door overlap handling."""
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    dist = float(np.linalg.norm(goal_pos - robot_pos))

    reward = 0
    overlap_info = check_robot_overlap(sim)
    
    # PRIMARY SIGNAL: Obstacle-based rewards
    overlap_type = overlap_info['overlap_type']
    if overlap_type == 'none':
        reward = -0.1
    elif overlap_type == 'person':
        reward = -1.0
    elif overlap_type == 'door':
        reward = -1.0  # ADD THIS: explicit door penalty
    elif overlap_type == 'both':
        reward = -2.0  # CHANGE: both should be worse than either alone
    else:
        reward = -0.1
    
    # ... rest of reward function
```

**Pros:** Quick fix, minimal changes
**Cons:** Doesn't address radius scaling issue

---

### Option B: Radius-Aware Reward (Recommended)

**Add clearance-based reward shaping:**

```python
def compute_reward(sim, progress_prev_dist: float, offset: float = 0.0) -> tuple[float, float, dict]:
    """
    Radius-aware reward with clearance gradient.
    
    Encourages maintaining clearance proportional to inflation radius:
    - Larger radius → need more clearance
    - Provides smooth gradient for learning
    """
    robot_pos = sim.robot.position
    goal_pos = sim.robot.goal
    dist = float(np.linalg.norm(goal_pos - robot_pos))

    reward = 0
    overlap_info = check_robot_overlap(sim)
    
    # PRIMARY SIGNAL: Obstacle-based rewards
    overlap_type = overlap_info['overlap_type']
    
    # Base penalties for overlaps
    if overlap_type == 'none':
        reward = -0.1
    elif overlap_type == 'person':
        reward = -1.0
    elif overlap_type == 'door':
        reward = -1.0
    elif overlap_type == 'both':
        reward = -2.0  # Both overlaps is worse
    else:
        reward = -0.1
    
    # ADDITIONAL: Radius-aware clearance reward for door
    if hasattr(sim.robot, 'door_position') and hasattr(sim.robot, 'corridor_bounds'):
        door_pos = np.array(sim.robot.door_position, dtype=float)
        door_radius = float(getattr(sim.robot.global_planner, 'door_halo_radius', 1.0))
        robot_radius = sim.robot.radius
        
        # Distance from robot center to door center
        dist_to_door = np.linalg.norm(robot_pos - door_pos)
        
        # Check if on inward-facing side (same logic as check_robot_overlap)
        bounds = sim.robot.corridor_bounds
        corridor_mid_y = (bounds['y_min'] + bounds['y_max']) * 0.5
        door_side = "left" if door_pos[1] < corridor_mid_y else "right"
        n_world = np.array([0.0, 1.0]) if door_side == "left" else np.array([0.0, -1.0])
        
        v_x = robot_pos[0] - door_pos[0]
        v_y = robot_pos[1] - door_pos[1]
        dot_product = n_world[0] * v_x + n_world[1] * v_y
        
        # Only apply clearance reward when on inward-facing side (near door)
        if dot_product > 0.0:
            # Clearance = actual distance - (door_radius + robot_radius)
            clearance = dist_to_door - (door_radius + robot_radius)
            
            # Desired clearance: safety margin proportional to door radius
            # Larger doors should have larger safety margins
            desired_clearance = 0.3 + 0.2 * door_radius  # 0.3m base + 20% of radius
            
            # Clearance reward/penalty (smooth gradient)
            if clearance < 0:
                # Inside inflation zone (already penalized above)
                clearance_reward = 0.0  # Don't double-penalize
            elif clearance < desired_clearance:
                # Too close but not overlapping
                # Linear penalty from -0.5 at boundary to 0 at desired clearance
                clearance_reward = -0.5 * (1.0 - clearance / desired_clearance)
            else:
                # Good clearance (beyond desired margin)
                clearance_reward = 0.0
            
            reward += clearance_reward
    
    # Rest of reward function (collisions, progress, etc.)
    # ...
    
    return reward, dist, info
```

**Pros:** 
- Radius-aware: larger radii automatically require more clearance
- Smooth gradient: helps learning
- Scales with curriculum difficulty

**Cons:** 
- More complex
- Requires tuning of `desired_clearance` formula

---

### Option C: Normalized Distance Reward (Advanced)

**Reward based on normalized clearance ratio:**

```python
# Inside the radius-aware section:
if dot_product > 0.0:
    # Normalize by door radius to make reward scale-invariant
    clearance = dist_to_door - (door_radius + robot_radius)
    normalized_clearance = clearance / door_radius
    
    # Reward based on normalized clearance
    # Small radius (0.8m): clearance=0.3m → normalized=0.375
    # Large radius (2.5m): clearance=0.3m → normalized=0.12
    # This teaches: "maintain clearance as fraction of radius"
    
    if normalized_clearance < 0:
        # Inside zone (already penalized)
        clearance_reward = 0.0
    elif normalized_clearance < 0.3:
        # Too close (< 30% of radius clearance)
        clearance_reward = -0.5 * (1.0 - normalized_clearance / 0.3)
    else:
        # Good clearance
        clearance_reward = 0.0
    
    reward += clearance_reward
```

**Pros:**
- Scale-invariant: teaches relative rather than absolute clearance
- Agent learns: "maintain 30% of radius as safety margin"
- Most aligned with curriculum learning goals

**Cons:**
- Most complex
- Hyperparameter (0.3) needs tuning

---

## Recommendation

**Start with Option A (Simple Fix)** and monitor training:

1. **Immediate fix**: Add missing door overlap case
2. **Train 50-100 episodes**: See if agent learns to avoid doors across radii
3. **If door collisions remain high**: Upgrade to Option B or C

### When to Use Each Option

| Scenario | Recommendation |
|----------|---------------|
| Agent learns well from binary signal | **Option A** - simple is better |
| High door collision rate across radii | **Option B** - need clearance gradient |
| Agent doesn't generalize to new radii | **Option C** - need scale-invariance |

---

## Implementation

### Quick Fix (Option A)

Replace the reward computation section in `compute_reward()`:

```python
# PRIMARY SIGNAL: Obstacle-based rewards
overlap_type = overlap_info['overlap_type']
if overlap_type == 'none':
    reward = -0.1
elif overlap_type == 'person':
    reward = -1.0
elif overlap_type == 'door':
    reward = -1.0  # ADDED: explicit door case
elif overlap_type == 'both':
    reward = -2.0  # CHANGED: worse than individual overlaps
else:
    reward = -0.1
```

### Full Implementation (Option B)

See the complete code in the "Option B" section above. Add after the overlap type checking.

---

## Testing the New Reward

After implementing, monitor these metrics:

1. **`overlap_door_pct` by radius**: Should be low across all radii
   ```python
   # In WandB, create custom charts:
   # X: door_halo_radius, Y: overlap_door_pct
   # Should show flat line (no correlation)
   ```

2. **Average clearance by radius**: Should scale with radius
   ```python
   # Add to logging:
   avg_clearance = np.mean(clearance_history)
   wandb.log({'avg_door_clearance': avg_clearance, 'door_halo_radius': door_radius})
   ```

3. **Success rate by radius bin**:
   - Small (0.8-1.3m): Should be >90%
   - Medium (1.3-2.0m): Should be >85%
   - Large (2.0-2.5m): Should be >80%

---

## Expected Training Behavior

### With Fixed Penalty (Option A)
- Agent learns: "avoid the inflation zone"
- May struggle with very large radii (harder to avoid)
- Performance might degrade for extreme sizes

### With Radius-Aware Reward (Option B/C)
- Agent learns: "maintain proportional clearance"
- Better generalization across curriculum stages
- More robust to radius variation

---

## Summary

**Critical Fix Required:**
```python
elif overlap_type == 'door':
    reward = -1.0  # This case is currently missing!
```

**Recommended Enhancement:**
Add radius-aware clearance reward (Option B) to encourage:
- Larger clearance for larger radii
- Smooth learning gradient
- Better generalization

**When to Apply:**
- Fix (Option A): Immediately before training
- Enhancement (Option B): If door collision rate > 15% after initial training
- Advanced (Option C): If agent doesn't generalize to unseen radii

The reward function is the **teacher** that tells the agent what's good behavior. With curriculum learning varying the environment, the reward should guide the agent to learn **generalizable** avoidance strategies!




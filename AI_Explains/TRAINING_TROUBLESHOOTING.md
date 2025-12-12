# Training Troubleshooting Guide

## Changes Made to Fix "No Learning" Issue

### Problem
Model wasn't learning after 100+ episodes - returns weren't improving.

### Root Cause
**Credit Assignment Problem**: The reward was only based on obstacles, with no connection between the agent's offset actions and their outcomes. The agent couldn't tell if its offsets helped or hurt.

---

## Solutions Implemented

### 1. **Improved Reward Function**

**Added back key components:**
```python
# Small progress reward (0.5x) - helps episodes complete
reward = 0.5 * progress

# Stronger obstacle penalties (primary signal)
- Free space: +0.5
- Person zone: -2.0 (increased from -1.0)
- Door zone: -1.0 (increased from -0.5)
- Both: -3.0

# Offset regularization (0.1x) - encourages purposeful use
reward -= 0.1 * abs(offset)

# Time penalty (-0.01) - encourages efficiency
reward -= 0.01

# Goal bonus (+20.0) - ensures successful completion
if dist < 1.0:
    reward += 20.0
```

**Why This Works:**
- Progress signal ensures agent knows episodes should end at goal
- Stronger obstacle penalties make avoidance signal clearer
- Offset regularization creates trade-off: "only deviate when necessary"
- Goal bonus motivates episode completion

### 2. **Increased Offset Range**
```python
max_offset = math.pi / 3  # ±60° (was ±30°)
```
More freedom to avoid obstacles.

### 3. **Better Logging**
Now tracks:
- Average absolute offset per episode
- Max absolute offset per episode
- Shows in degrees for interpretability

**Example output:**
```
Episode 1/50 | Return: 45.23 | Steps: 342
  Overlaps - Free: 85.4% | Person: 12.1% | Door: 2.5% | Both: 0.0%
  Offsets - Avg: 8.3° | Max: 42.1° | (range: ±60°)
```

---

## What to Watch During Training

### ✅ **Good Signs (Learning is Working)**

1. **Returns increasing over time**
   - Episode 1-10: ~20-40
   - Episode 50-100: ~60-100
   - Trending upward

2. **Overlap percentages improving**
   - Free space % increasing (towards 90%+)
   - Person overlap % decreasing (towards <10%)
   - Door overlap % decreasing

3. **Offsets being used purposefully**
   - Early episodes: High variance, random (avg ~20-30°)
   - Later episodes: Lower average (avg ~5-15°), but spikes when needed
   - Pattern: Small offsets normally, large offsets near obstacles

4. **Episode lengths stable or increasing**
   - Early: Might end in collision (short episodes)
   - Later: Reaching goal consistently (longer episodes)

### ❌ **Bad Signs (Still Not Learning)**

1. **Returns flat or decreasing**
   - All episodes ~20-30 with no improvement
   - High variance with no trend

2. **Offsets all near zero OR all maxed out**
   - All ~0°: Agent not using offsets (path alone is better)
   - All ~60°: Agent always maxing out (not learning nuance)

3. **High collision rate**
   - Many episodes ending early
   - Collision count not decreasing

4. **Overlap percentages not improving**
   - Person overlap staying high (>30%)
   - No clear trend

---

## Hyperparameter Tuning

If still not learning, try adjusting:

### **Learning Rates**
```bash
# Current default: 1e-3 for both
python train.py --lr-actor 3e-4 --lr-critic 1e-3  # Lower actor LR

# Or higher:
python train.py --lr-actor 3e-3 --lr-critic 3e-3  # Faster but less stable
```

### **Exploration (Action Std)**
```python
# In train.py line 336-337, change:
action_std_init=0.6,  # Current: 0.4, try higher for more exploration
```

### **Obstacle Penalties**
```python
# In compute_reward(), increase if agent isn't avoiding:
elif overlap_type == 'person':
    reward += -5.0  # Was -2.0, make even stronger
```

### **Update Frequency**
```bash
# Current: updates every 2000 steps
python train.py --update-timestep 1000  # More frequent updates
```

### **Episode Length**
```bash
# Current: 800 max steps
python train.py --max-steps 500  # Shorter episodes = faster learning cycles
```

---

## Quick Diagnosis Commands

### **Check if agent is actually using offsets:**
```bash
python train.py --episodes 10 | grep "Offsets"
```
If all offsets are ~0°, agent isn't learning to use them.

### **Watch training live:**
```bash
python train.py --use-wandb --episodes 100
# Open wandb link to see real-time charts
```

### **Test baseline (no agent):**
Set all offsets to 0 and see baseline performance:
```python
# In train.py, temporarily set:
offset = 0.0  # Instead of agent output
```
If baseline performs similarly to trained agent, agent isn't helping.

---

## Expected Learning Curve

### **Episodes 1-20: Exploration**
- Returns: 20-50
- High variance
- Random offsets
- Many collisions

### **Episodes 20-50: Discovery**
- Returns: 40-80
- Starting to avoid obstacles
- Offsets becoming more purposeful
- Fewer collisions

### **Episodes 50-100: Refinement**
- Returns: 60-120
- Consistent obstacle avoidance
- Small offsets in free space, large near obstacles
- Rare collisions

### **Episodes 100+: Mastery**
- Returns: 80-150
- >90% free space
- Efficient navigation
- Minimal unnecessary offsets

---

## Common Issues & Fixes

### **Issue: Returns stuck around 20-30**
**Diagnosis**: Agent not reaching goal  
**Fix**: Increase progress reward weight
```python
reward = 1.0 * progress  # Increase from 0.5
```

### **Issue: High collision rate (>20%)**
**Diagnosis**: Collision penalty not strong enough  
**Fix**: Increase collision penalty
```python
reward += -20.0  # Increase from -10.0
```

### **Issue: Offsets always 0°**
**Diagnosis**: 
1. Path alone is good enough (agent not needed), OR
2. Offset regularization too strong

**Fix**: Reduce offset penalty
```python
offset_penalty = 0.01 * abs(offset)  # Reduce from 0.1
```

### **Issue: Offsets always maxed at ±60°**
**Diagnosis**: Agent learning to always turn hard  
**Fix**: 
1. Increase offset regularization
2. Check if obstacles are too difficult
3. Verify state features include goal direction

### **Issue: Unstable training (returns jumping wildly)**
**Diagnosis**: Learning rate too high or batch size too small  
**Fix**:
```bash
python train.py --lr-actor 1e-4 --lr-critic 3e-4
```

---

## Validation Checklist

After training, verify the agent learned by testing:

1. **Render a test episode:**
```bash
python test.py --render --episodes 1
```
Watch: Does robot smoothly avoid obstacles?

2. **Check offset behavior:**
```bash
python test.py --episodes 3
```
Look at printed offsets - Do they make sense?

3. **Compare to baseline:**
Run same scenario with offset=0 (pure path following)
Agent should perform better.

4. **Stress test:**
Increase number of people or reduce corridor width
Agent should adapt offsets accordingly.

---

## Contact Points for Further Debugging

If still not working after trying above:

1. **Share wandb link** - We can look at training curves
2. **Print sample states** - Check if features make sense
3. **Verify simulator** - Ensure obstacles are actually present
4. **Check path planner** - Ensure it's not already perfect (making agent redundant)


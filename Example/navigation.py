import math
import copy

class NavigationAlgorithm:
    """Dynamic Window Approach (DWA) navigation algorithm."""
    
    def __init__(self, max_velocity=0.5, max_acceleration=0.5, dt=0.1, 
                 steps_ahead=20, forward_weight=12, obstacle_weight=6666):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.dt = dt
        self.steps_ahead = steps_ahead
        self.tau = dt * steps_ahead
        self.forward_weight = forward_weight
        self.obstacle_weight = obstacle_weight
        self.safe_dist = 0.1  # Safe distance for collision avoidance
        
        # Robot width (distance between wheels)
        self.W = 0.2  # 2 * robot_radius
        
    def predict_position(self, vL, vR, x, y, theta, deltat):
        """Predict new robot position based on current pose and velocity controls."""
        # Simple special cases
        # Straight line motion
        if round(vL, 3) == round(vR, 3):
            xnew = x + vL * deltat * math.cos(theta)
            ynew = y + vL * deltat * math.sin(theta)
            thetanew = theta
            path = (0, vL * deltat)   # 0 indicates pure translation
        # Pure rotation motion
        elif round(vL, 3) == -round(vR, 3):
            xnew = x
            ynew = y
            thetanew = theta + ((vR - vL) * deltat / self.W)
            path = (1, 0) # 1 indicates pure rotation
        else:
            # Rotation and arc angle of general circular motion
            R = self.W / 2.0 * (vR + vL) / (vR - vL)
            deltatheta = (vR - vL) * deltat / self.W
            xnew = x + R * (math.sin(deltatheta + theta) - math.sin(theta))
            ynew = y - R * (math.cos(deltatheta + theta) - math.cos(theta))
            thetanew = theta + deltatheta

            # To calculate parameters for arc drawing
            (cx, cy) = (x - R * math.sin(theta), y + R * math.cos(theta))
            Rabs = abs(R)
            ((tlx, tly), (Rx, Ry)) = ((cx - Rabs, cy + Rabs), (2 * Rabs, 2 * Rabs))
            if (R > 0):
                start_angle = theta - math.pi/2.0
            else:
                start_angle = theta + math.pi/2.0
            stop_angle = start_angle + deltatheta
            path = (2, ((tlx, tly), (Rx, Ry)), start_angle, stop_angle)

        return (xnew, ynew, thetanew, path)
    
    def calculate_closest_obstacle_distance(self, x, y, obstacles, target_index):
        """Calculate the closest obstacle distance at a position."""
        closest_dist = 100000.0
        for i, obstacle in enumerate(obstacles):
            if i != target_index:
                dist = obstacle.distance_to(x, y)
                # Distance between closest touching point of circular robot and circular barrier
                dist = dist - obstacle.radius - 0.1  # 0.1 is robot radius
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist
    
    def plan(self, x, y, theta, vL, vR, obstacles, target_index):
        """Plan the best velocity commands using DWA."""
        best_benefit = -100000
        best_index = 0  # Track which trajectory was the best
        
        # Copy of obstacles so we can predict their positions
        obstacles_copy = copy.deepcopy(obstacles)
        
        # Predict obstacle positions
        for i in range(self.steps_ahead):
            for obstacle in obstacles_copy:
                obstacle.update(self.dt, (-4.0, -3.0, 4.0, 3.0))  # PLAYFIELDCORNERS
        
        # Range of possible motions: generate more samples for better visualization
        # Create more velocity samples to see more trajectories
        vL_min = max(-self.max_velocity, vL - self.max_acceleration * self.dt)
        vL_max = min(self.max_velocity, vL + self.max_acceleration * self.dt)
        vR_min = max(-self.max_velocity, vR - self.max_acceleration * self.dt)
        vR_max = min(self.max_velocity, vR + self.max_acceleration * self.dt)
        
        # Generate 5 samples for each velocity component
        vL_possible_array = [vL_min + i * (vL_max - vL_min) / 4 for i in range(5)]
        vR_possible_array = [vR_min + i * (vR_max - vR_min) / 4 for i in range(5)]
        
        paths_to_draw = []
        new_positions_to_draw = []
        trajectory_index = 0
        
        for vL_possible in vL_possible_array:
            for vR_possible in vR_possible_array:
                # Velocity limits are already enforced when creating the arrays
                
                # Predict new position in TAU seconds
                (xpredict, ypredict, thetapredict, path) = self.predict_position(
                    vL_possible, vR_possible, x, y, theta, self.tau)
                paths_to_draw.append(path)
                new_positions_to_draw.append((xpredict, ypredict))
                
                # What is the distance to the closest obstacle from this possible position?
                distance_to_obstacle = self.calculate_closest_obstacle_distance(
                    xpredict, ypredict, obstacles_copy, target_index)
                
                # Calculate how much closer we've moved to target location
                target_pos = obstacles[target_index].get_position()
                previous_target_distance = math.sqrt((x - target_pos[0])**2 + (y - target_pos[1])**2)
                new_target_distance = math.sqrt((xpredict - target_pos[0])**2 + (ypredict - target_pos[1])**2)
                distance_forward = previous_target_distance - new_target_distance
                
                # Positive benefit
                distance_benefit = self.forward_weight * distance_forward
                
                # Negative cost: once we are less than SAFEDIST from collision, linearly increasing cost
                if distance_to_obstacle < self.safe_dist:
                    obstacle_cost = self.obstacle_weight * (self.safe_dist - distance_to_obstacle)
                else:
                    obstacle_cost = 0.0
                
                # Total benefit function to optimise
                benefit = distance_benefit - obstacle_cost
                if benefit > best_benefit:
                    vL_chosen = vL_possible
                    vR_chosen = vR_possible
                    best_benefit = benefit
                    best_index = trajectory_index
                
                trajectory_index += 1
        
        return vL_chosen, vR_chosen, paths_to_draw, new_positions_to_draw, best_index 
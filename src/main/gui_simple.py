import tkinter as tk
from tkinter import ttk, messagebox
import pygame
import time
import sys
import os
import random
import threading

# Add the parent directory to the path to import simulation modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.sim import Simulation


class SimpleSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictive DWA Simulation (Embedded)")
        self.root.geometry("1200x800")
        
        # Simulation state
        self.simulation = None
        self.running = False
        self.pygame_screen = None
        
        # Create GUI layout
        self.create_widgets()
        
        # Initialize simulation
        self.initialize_simulation()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel for simulation display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control buttons
        self.create_control_buttons(left_panel)
        
        # DWA Parameters
        self.create_dwa_parameters(left_panel)
        
        # Simulation Parameters
        self.create_simulation_parameters(left_panel)
        
        # Simulation display
        self.create_simulation_display(right_panel)
        
    def create_control_buttons(self, parent):
        # Control section
        control_frame = ttk.LabelFrame(parent, text="Simulation Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Button frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        # Run button
        self.run_button = ttk.Button(button_frame, text="Run", command=self.start_simulation)
        self.run_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Stop button
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Restart button
        self.restart_button = ttk.Button(button_frame, text="Restart", command=self.restart_simulation)
        self.restart_button.pack(side=tk.LEFT)
        
    def create_dwa_parameters(self, parent):
        # DWA Parameters section
        dwa_frame = ttk.LabelFrame(parent, text="DWA Parameters", padding=10)
        dwa_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Parameter variables
        self.dwa_params = {
            'max_speed': tk.DoubleVar(value=2.0),
            'max_rotation': tk.DoubleVar(value=3.14159),  # π
            'max_accel': tk.DoubleVar(value=4.0),
            'max_angular_accel': tk.DoubleVar(value=6.28318),  # 2π
            'predict_time': tk.DoubleVar(value=2.0),
            'v_samples': tk.IntVar(value=8),
            'w_samples': tk.IntVar(value=8),
            'goal_weight': tk.DoubleVar(value=0.6),
            'clearance_weight': tk.DoubleVar(value=0.3),
            'velocity_weight': tk.DoubleVar(value=0.1)
        }
        
        # Create parameter controls
        row = 0
        for param_name, var in self.dwa_params.items():
            # Convert parameter name to display name
            display_name = param_name.replace('_', ' ').title()
            
            # Label
            label = ttk.Label(dwa_frame, text=f"{display_name}:")
            label.grid(row=row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            
            # Entry/Scale based on parameter type
            if isinstance(var, tk.IntVar):
                # Use Spinbox for integers
                widget = ttk.Spinbox(dwa_frame, from_=1, to=20, textvariable=var, width=10)
            else:
                # Use Scale for doubles with appropriate ranges
                if param_name == 'max_speed':
                    widget = ttk.Scale(dwa_frame, from_=0.5, to=5.0, variable=var, orient=tk.HORIZONTAL)
                elif param_name == 'max_rotation':
                    widget = ttk.Scale(dwa_frame, from_=1.0, to=6.0, variable=var, orient=tk.HORIZONTAL)
                elif param_name == 'max_accel':
                    widget = ttk.Scale(dwa_frame, from_=1.0, to=10.0, variable=var, orient=tk.HORIZONTAL)
                elif param_name == 'max_angular_accel':
                    widget = ttk.Scale(dwa_frame, from_=2.0, to=12.0, variable=var, orient=tk.HORIZONTAL)
                elif param_name == 'predict_time':
                    widget = ttk.Scale(dwa_frame, from_=0.5, to=5.0, variable=var, orient=tk.HORIZONTAL)
                elif param_name in ['goal_weight', 'clearance_weight', 'velocity_weight']:
                    widget = ttk.Scale(dwa_frame, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL)
                else:
                    widget = ttk.Scale(dwa_frame, from_=0.1, to=10.0, variable=var, orient=tk.HORIZONTAL)
            
            widget.grid(row=row, column=1, sticky=tk.EW, padx=(0, 5), pady=2)
            
            # Create formatted display variable
            display_var = tk.StringVar()
            display_var.set(f"{var.get():.2f}" if isinstance(var, tk.DoubleVar) else str(var.get()))
            
            # Value label with formatted display
            value_label = ttk.Label(dwa_frame, textvariable=display_var)
            value_label.grid(row=row, column=2, sticky=tk.W, pady=2)
            
            # Bind the original variable to update the display
            def update_display(var=var, display_var=display_var):
                if isinstance(var, tk.DoubleVar):
                    display_var.set(f"{var.get():.2f}")
                else:
                    display_var.set(str(var.get()))
            
            var.trace('w', lambda *args, v=var, d=display_var: update_display(v, d))
            
            row += 1
        
        # Apply button
        apply_button = ttk.Button(dwa_frame, text="Apply DWA Parameters", command=self.apply_dwa_parameters)
        apply_button.grid(row=row, column=0, columnspan=3, pady=(10, 0))
        
        # Configure grid weights
        dwa_frame.columnconfigure(1, weight=1)
        
    def create_simulation_parameters(self, parent):
        # Simulation Parameters section
        sim_frame = ttk.LabelFrame(parent, text="Simulation Parameters", padding=10)
        sim_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Parameter variables
        self.sim_params = {
            'corridor_width': tk.DoubleVar(value=4.0),
            'num_people': tk.IntVar(value=3),
            'spawn_interval': tk.DoubleVar(value=1.0),
            'door_side': tk.StringVar(value="right")
        }
        
        # Create parameter controls
        row = 0
        for param_name, var in self.sim_params.items():
            # Convert parameter name to display name
            display_name = param_name.replace('_', ' ').title()
            
            # Label
            label = ttk.Label(sim_frame, text=f"{display_name}:")
            label.grid(row=row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            
            # Widget based on parameter type
            if param_name == 'door_side':
                # Use Combobox for door side
                widget = ttk.Combobox(sim_frame, textvariable=var, values=["left", "right"], state="readonly", width=8)
            elif isinstance(var, tk.IntVar):
                # Use Spinbox for integers
                widget = ttk.Spinbox(sim_frame, from_=1, to=10, textvariable=var, width=8)
            else:
                # Use Scale for doubles
                if param_name == 'corridor_width':
                    widget = ttk.Scale(sim_frame, from_=2.0, to=8.0, variable=var, orient=tk.HORIZONTAL)
                elif param_name == 'spawn_interval':
                    widget = ttk.Scale(sim_frame, from_=0.5, to=3.0, variable=var, orient=tk.HORIZONTAL)
                else:
                    widget = ttk.Scale(sim_frame, from_=0.1, to=10.0, variable=var, orient=tk.HORIZONTAL)
            
            widget.grid(row=row, column=1, sticky=tk.EW, padx=(0, 5), pady=2)
            
            # Create formatted display variable
            display_var = tk.StringVar()
            display_var.set(f"{var.get():.2f}" if isinstance(var, tk.DoubleVar) else str(var.get()))
            
            # Value label with formatted display
            value_label = ttk.Label(sim_frame, textvariable=display_var)
            value_label.grid(row=row, column=2, sticky=tk.W, pady=2)
            
            # Bind the original variable to update the display
            def update_display(var=var, display_var=display_var):
                if isinstance(var, tk.DoubleVar):
                    display_var.set(f"{var.get():.2f}")
                else:
                    display_var.set(str(var.get()))
            
            var.trace('w', lambda *args, v=var, d=display_var: update_display(v, d))
            
            row += 1
        
        # Apply button
        apply_button = ttk.Button(sim_frame, text="Apply Simulation Parameters", command=self.apply_simulation_parameters)
        apply_button.grid(row=row, column=0, columnspan=3, pady=(10, 0))
        
        # Configure grid weights
        sim_frame.columnconfigure(1, weight=1)
        
    def create_simulation_display(self, parent):
        # Simulation display section
        display_frame = ttk.LabelFrame(parent, text="Simulation Display", padding=10)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for pygame surface
        self.canvas = tk.Canvas(display_frame, bg='white', width=800, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind canvas resize
        self.canvas.bind('<Configure>', self.on_canvas_resize)
        
    def on_canvas_resize(self, event):
        # Handle canvas resize if needed
        pass
        
    def initialize_simulation(self):
        """Initialize the simulation with GUI parameters"""
        try:
            # Get parameters from GUI (or use defaults if not available)
            if hasattr(self, 'sim_params'):
                corridor_width = self.sim_params['corridor_width'].get()
                door_side = self.sim_params['door_side'].get()
                num_people = self.sim_params['num_people'].get()
            else:
                corridor_width = 4.0
                door_side = "right"
                num_people = 3
            
            # Generate random speeds for people
            people_speeds = [random.uniform(1.0, 1.5) for _ in range(num_people)]
            
            self.simulation = Simulation(
                corridor_width=corridor_width,
                door_side=door_side,
                num_people=num_people,
                people_speeds=people_speeds
            )
            self.apply_dwa_parameters()  # Apply initial DWA parameters
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize simulation: {str(e)}")
            
    def apply_dwa_parameters(self):
        """Apply DWA parameters to the current simulation"""
        if self.simulation is None or not hasattr(self.simulation.robot, 'nav'):
            return
            
        try:
            nav = self.simulation.robot.nav
            
            # Apply basic parameters
            nav.max_speed = self.dwa_params['max_speed'].get()
            nav.max_rotation = self.dwa_params['max_rotation'].get()
            nav.max_accel = self.dwa_params['max_accel'].get()
            nav.max_angular_accel = self.dwa_params['max_angular_accel'].get()
            nav.predict_time = self.dwa_params['predict_time'].get()
            nav.v_samples = self.dwa_params['v_samples'].get()
            nav.w_samples = self.dwa_params['w_samples'].get()
            
            # Apply scoring weights
            nav.weights['goal'] = self.dwa_params['goal_weight'].get()
            nav.weights['clearance'] = self.dwa_params['clearance_weight'].get()
            nav.weights['velocity'] = self.dwa_params['velocity_weight'].get()
            
            # Note: safe_distance is hardcoded in clearance_score methods
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply DWA parameters: {str(e)}")
            
    def apply_simulation_parameters(self):
        """Apply simulation parameters and restart if running"""
        if self.simulation is None:
            return
            
        try:
            # Store current state
            was_running = self.running
            
            # Stop simulation if running
            if was_running:
                self.stop_simulation()
                time.sleep(0.1)
            
            # Reinitialize simulation with new parameters
            self.initialize_simulation()
            
            # Restart if it was running
            if was_running:
                self.start_simulation()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply simulation parameters: {str(e)}")
            
    def start_simulation(self):
        """Start the simulation in embedded mode"""
        if self.running:
            return
            
        print("Starting simulation...")
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start simulation in a separate thread
        self.simulation_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.simulation_thread.start()
        
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Wait for thread to finish
        if hasattr(self, 'simulation_thread') and self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
        
    def restart_simulation(self):
        """Restart the simulation"""
        self.stop_simulation()
        time.sleep(0.1)  # Brief pause to ensure thread stops
        
        # Reinitialize simulation
        self.initialize_simulation()
        
    def simulation_loop(self):
        """Main simulation loop running in separate thread"""
        try:
            print("Initializing pygame...")
            # Initialize pygame without display
            pygame.init()
            
            # Create a surface for rendering
            width, height = 800, 400
            self.pygame_screen = pygame.Surface((width, height))
            clock = pygame.time.Clock()
            print("Pygame initialized successfully")
            
            while self.running:
                dt = clock.tick(60) / 1000.0  # Delta time in seconds
                
                # Update simulation
                if self.simulation:
                    try:
                        state, reward, done = self.simulation.step(dt)
                        
                        # Check if simulation is done
                        if done:
                            self.running = False
                            break
                    except Exception as e:
                        print(f"Simulation step error: {e}")
                        self.running = False
                        break
                
                # Render to surface
                self.pygame_screen.fill((255, 255, 255))
                if self.simulation:
                    try:
                        self.simulation.draw_v0(self.pygame_screen)
                    except Exception as e:
                        print(f"Rendering error: {e}")
                
                # Update the canvas in the main thread
                self.root.after(0, self.update_canvas)
                
        except Exception as e:
            print(f"Simulation loop error: {e}")
            self.running = False
        finally:
            try:
                pygame.quit()
            except:
                pass
            
        # Update GUI state when simulation ends
        self.root.after(0, self.stop_simulation)
    
    def update_canvas(self):
        """Update the tkinter canvas with the pygame surface"""
        if not hasattr(self, 'pygame_screen') or self.pygame_screen is None:
            return
            
        try:
            # Convert pygame surface to PIL Image
            import PIL.Image
            import PIL.ImageTk
            
            # Get the pygame surface data
            pygame_string = pygame.image.tostring(self.pygame_screen, 'RGB')
            
            # Create PIL Image
            width, height = self.pygame_screen.get_size()
            pil_image = PIL.Image.frombytes('RGB', (width, height), pygame_string)
            
            # Convert to PhotoImage
            photo = PIL.ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Canvas update error: {e}")
            
    def on_closing(self):
        """Handle application closing"""
        self.stop_simulation()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SimpleSimulationGUI(root)
    
    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main() 
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import time
import random
import threading
import queue

# Add the parent directory to the path to import simulation modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.sim import Simulation


class MacOSFixedSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictive DWA Simulation (macOS Fixed)")
        self.root.geometry("1200x800")
        
        # Simulation state
        self.simulation = None
        self.running = False
        self.simulation_process = None
        self.status_queue = queue.Queue()
        
        # Create GUI layout
        self.create_widgets()
        
        # Initialize simulation
        self.initialize_simulation()
        
        # Start status checker
        self.check_status()
        
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
            
            # Value label
            value_label = ttk.Label(dwa_frame, textvariable=var)
            value_label.grid(row=row, column=2, sticky=tk.W, pady=2)
            
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
            
            # Value label
            value_label = ttk.Label(sim_frame, textvariable=var)
            value_label.grid(row=row, column=2, sticky=tk.W, pady=2)
            
            row += 1
        
        # Apply button
        apply_button = ttk.Button(sim_frame, text="Apply Simulation Parameters", command=self.apply_simulation_parameters)
        apply_button.grid(row=row, column=0, columnspan=3, pady=(10, 0))
        
        # Configure grid weights
        sim_frame.columnconfigure(1, weight=1)
        
    def create_simulation_display(self, parent):
        # Simulation display section
        display_frame = ttk.LabelFrame(parent, text="Simulation Status", padding=10)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = ttk.Label(display_frame, text="Simulation not running", font=("Arial", 12))
        self.status_label.pack(pady=20)
        
        # Info text
        info_text = """
        This GUI controls the simulation parameters.
        
        When you click "Run", a separate pygame window will open
        showing the simulation.
        
        You can adjust parameters in this GUI and they will be
        applied to the simulation when you restart it.
        
        Note: On macOS, the simulation window may appear behind
        other windows. Check your dock or use Cmd+Tab to find it.
        """
        
        info_label = ttk.Label(display_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(pady=20)
        
        # Console output
        console_frame = ttk.LabelFrame(display_frame, text="Console Output", padding=5)
        console_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.console_text = tk.Text(console_frame, height=10, width=50)
        scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.console_text.yview)
        self.console_text.configure(yscrollcommand=scrollbar.set)
        
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def log_message(self, message):
        """Add a message to the console"""
        self.console_text.insert(tk.END, f"{message}\n")
        self.console_text.see(tk.END)
        
    def check_status(self):
        """Check simulation status periodically"""
        try:
            while not self.status_queue.empty():
                status = self.status_queue.get_nowait()
                if status == "started":
                    self.status_label.config(text="Simulation running")
                    self.log_message("Simulation started successfully")
                elif status == "stopped":
                    self.status_label.config(text="Simulation stopped")
                    self.log_message("Simulation stopped")
                elif status == "error":
                    self.status_label.config(text="Simulation error")
                    self.log_message("Simulation encountered an error")
        except queue.Empty:
            pass
        
        # Check if process is still running
        if self.simulation_process and self.running:
            if self.simulation_process.poll() is not None:
                # Process has ended
                self.running = False
                self.run_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.status_label.config(text="Simulation ended")
                self.log_message("Simulation process ended")
                self.simulation_process = None
        
        # Schedule next check
        self.root.after(100, self.check_status)
        
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
        """Start the simulation in a separate subprocess"""
        if self.running:
            return
            
        self.log_message("Starting simulation...")
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start simulation in a separate process
        try:
            # Create a temporary script to run the simulation
            script_content = f'''
import sys
import os
import time

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from sim.sim import Simulation
import pygame
import random

print("Initializing simulation...")

# Create simulation with current parameters
sim = Simulation(
    corridor_width={self.sim_params['corridor_width'].get()},
    door_side="{self.sim_params['door_side'].get()}",
    num_people={self.sim_params['num_people'].get()},
    people_speeds=[random.uniform(1.0, 1.5) for _ in range({self.sim_params['num_people'].get()})]
)

# Apply DWA parameters
nav = sim.robot.nav
nav.max_speed = {self.dwa_params['max_speed'].get()}
nav.max_rotation = {self.dwa_params['max_rotation'].get()}
nav.max_accel = {self.dwa_params['max_accel'].get()}
nav.max_angular_accel = {self.dwa_params['max_angular_accel'].get()}
nav.predict_time = {self.dwa_params['predict_time'].get()}
nav.v_samples = {self.dwa_params['v_samples'].get()}
nav.w_samples = {self.dwa_params['w_samples'].get()}
nav.weights['goal'] = {self.dwa_params['goal_weight'].get()}
nav.weights['clearance'] = {self.dwa_params['clearance_weight'].get()}
nav.weights['velocity'] = {self.dwa_params['velocity_weight'].get()}

print("Starting pygame...")

# Run simulation with macOS-specific fixes
os.environ['SDL_VIDEODRIVER'] = 'x11'  # Force X11 driver on macOS
pygame.init()

# Set display mode with specific flags for macOS
screen = pygame.display.set_mode((800, 400), pygame.DOUBLEBUF | pygame.HWSURFACE)
pygame.display.set_caption("Predictive DWA Simulation")
clock = pygame.time.Clock()

print("Simulation window opened successfully")

# Force a display update to prevent black screen
screen.fill((255, 255, 255))
pygame.display.flip()
time.sleep(0.1)  # Brief pause to ensure display is ready

running = True
while running:
    dt = clock.tick(60) / 1000.0
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    
    try:
        state, reward, done = sim.step(dt)
        if done:
            running = False
            break
    except Exception as e:
        print(f"Simulation error: {{e}}")
        running = False
        break
    
    screen.fill((255, 255, 255))
    sim.draw_v0(screen)
    pygame.display.flip()

print("Simulation ended")
pygame.quit()
'''
            
            # Write the script to a temporary file
            script_path = "temp_simulation_macos_fixed.py"
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Start the simulation process with environment variables
            env = os.environ.copy()
            env['SDL_VIDEODRIVER'] = 'x11'  # Force X11 driver
            env['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hide pygame support prompt
            
            self.simulation_process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            self.log_message("Simulation process started")
            self.status_queue.put("started")
            
        except Exception as e:
            self.log_message(f"Failed to start simulation: {e}")
            self.running = False
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_queue.put("error")
        
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log_message("Stopping simulation...")
        
        # Terminate the subprocess if it's running
        if self.simulation_process:
            try:
                self.simulation_process.terminate()
                self.simulation_process.wait(timeout=5)
                self.log_message("Simulation process terminated")
            except:
                try:
                    self.simulation_process.kill()
                    self.log_message("Simulation process killed")
                except:
                    pass
            self.simulation_process = None
        
        self.status_queue.put("stopped")
        
    def restart_simulation(self):
        """Restart the simulation"""
        self.stop_simulation()
        time.sleep(0.1)  # Brief pause to ensure process stops
        
        # Reinitialize simulation
        self.initialize_simulation()
        
    def on_closing(self):
        """Handle application closing"""
        self.stop_simulation()
        
        # Clean up temporary file
        try:
            if os.path.exists("temp_simulation_macos_fixed.py"):
                os.remove("temp_simulation_macos_fixed.py")
        except:
            pass
            
        self.root.destroy()


def main():
    root = tk.Tk()
    app = MacOSFixedSimulationGUI(root)
    
    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main() 
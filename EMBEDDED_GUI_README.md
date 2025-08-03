# Embedded GUI Fix

## Problem
The original GUI implementations opened the simulation in a separate pygame window when the "Run" button was pressed, instead of displaying it within the GUI itself.

## Solution
I've modified the `gui_simple.py` to embed the simulation within the tkinter canvas. The key changes are:

### 1. Threading Implementation
- The simulation now runs in a separate thread to avoid blocking the GUI
- The pygame surface is rendered to a `pygame.Surface` instead of a display window
- The surface is converted to a PIL Image and displayed in the tkinter canvas

### 2. Canvas Integration
- The simulation display area now contains a tkinter canvas
- The pygame surface is converted to a PIL Image using `pygame.image.tostring()`
- The PIL Image is converted to a tkinter PhotoImage and displayed on the canvas

### 3. Real-time Updates
- The canvas is updated in real-time using `root.after(0, self.update_canvas)`
- This ensures the simulation runs smoothly without blocking the GUI

## Files Modified

### `src/main/gui.py` (renamed from `gui_simple.py`)
- Added threading support
- Modified `simulation_loop()` to render to a surface instead of display
- Added `update_canvas()` method to convert pygame surface to tkinter canvas
- Updated `start_simulation()` to run in a separate thread
- Updated `stop_simulation()` to properly handle thread cleanup
- **Fixed decimal display**: All values now show maximum 2 decimal places

### `run_gui.py` (renamed from `run_gui_simple.py`)
- Updated description to reflect embedded functionality

### `requirements.txt`
- Added `Pillow>=8.0.0` for image conversion

## How to Use

### Prerequisites
Install the required dependencies:
```bash
pip install Pillow
```

### Running the Embedded GUI
```bash
python run_gui.py
```

### Features
- **Embedded Simulation**: The simulation now runs within the GUI canvas instead of a separate window
- **Real-time Controls**: All parameter controls work in real-time
- **Threading**: The simulation runs in a background thread, keeping the GUI responsive
- **Smooth Updates**: The canvas updates at 60 FPS for smooth animation
- **Clean Display**: All decimal values are formatted to show maximum 2 decimal places

## Technical Details

### Pygame Surface to Tkinter Canvas Conversion
```python
# Convert pygame surface to PIL Image
pygame_string = pygame.image.tostring(self.pygame_screen, 'RGB')
pil_image = PIL.Image.frombytes('RGB', (width, height), pygame_string)

# Convert to PhotoImage for tkinter
photo = PIL.ImageTk.PhotoImage(pil_image)

# Update canvas
self.canvas.delete("all")
self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
self.canvas.image = photo  # Keep a reference
```

### Threading Architecture
- Main thread: Handles GUI events and canvas updates
- Simulation thread: Runs the simulation loop and renders to pygame surface
- Communication: Uses `root.after()` to safely update the GUI from the simulation thread

## Benefits
1. **Unified Interface**: Everything is contained within one window
2. **Better UX**: No separate windows to manage
3. **Responsive Controls**: GUI remains responsive during simulation
4. **Real-time Parameter Adjustment**: Parameters can be changed while simulation is running

## Troubleshooting

### PIL/Pillow Not Found
If you get an error about PIL not being found:
```bash
pip install Pillow
```

### Performance Issues
If the simulation runs slowly:
- Reduce the canvas size
- Lower the frame rate in the simulation loop
- Check that your system has sufficient resources

### Threading Issues
If you experience threading-related errors:
- Make sure to use `root.after()` for GUI updates
- Always check if the thread is alive before joining
- Use daemon threads to ensure they terminate with the main program 
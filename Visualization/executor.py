import pygame
import pygame.font
import sys
import subprocess
import os

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 450
BACKGROUND_COLOR = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)
FIELD_COLOR = (60, 60, 60)
FIELD_ACTIVE_COLOR = (80, 80, 80)
BUTTON_COLOR = (80, 120, 130)
BUTTON_HOVER_COLOR = (100, 140, 150)
BORDER_COLOR = (100, 100, 100)
TITLE_COLOR = (200, 200, 200)

# Create the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("N-Queens Genetic Algorithm Executor")

# Font initialization
title_font = pygame.font.Font(None, 36)
label_font = pygame.font.Font(None, 28)
input_font = pygame.font.Font(None, 28)
button_font = pygame.font.Font(None, 32)

# Input field class
class InputField:
    def __init__(self, x, y, width, height, default_value, numeric_only=True, min_value=None, max_value=None, decimal=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = FIELD_COLOR
        self.text = str(default_value)
        self.default_value = str(default_value)
        self.active = False
        self.numeric_only = numeric_only
        self.min_value = min_value
        self.max_value = max_value
        self.decimal = decimal
        self.cursor_visible = True
        self.cursor_timer = 0
        self.rendered_text = input_font.render(self.text, True, TEXT_COLOR)
        self.text_rect = self.rendered_text.get_rect(midleft=(x + 10, y + height//2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle active state based on mouse click
            self.active = self.rect.collidepoint(event.pos)
            self.color = FIELD_ACTIVE_COLOR if self.active else FIELD_COLOR
            
            # If field is deactivated, validate the input
            if not self.active and self.text:
                self.validate_input()
        
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN or event.key == pygame.K_TAB:
                self.active = False
                self.color = FIELD_COLOR
                # Validate when confirming with Enter/Tab
                if self.text:
                    self.validate_input()
            else:
                if self.numeric_only:
                    # Only allow numeric input (and decimal point if enabled)
                    if event.unicode.isdigit() or (event.unicode == '.' and self.decimal and '.' not in self.text):
                        self.text += event.unicode
                else:
                    # Allow any character
                    self.text += event.unicode
            
            # Update the rendered text
            self.rendered_text = input_font.render(self.text, True, TEXT_COLOR)
            self.text_rect = self.rendered_text.get_rect(midleft=(self.rect.x + 10, self.rect.y + self.rect.height//2))
    
    def validate_input(self):
        """Validate the input value against min/max constraints"""
        try:
            value = float(self.text) if self.decimal else int(self.text)
            if self.min_value is not None and value < self.min_value:
                self.text = str(self.min_value)
            elif self.max_value is not None and value > self.max_value:
                self.text = str(self.max_value)
            # Update rendered text after validation
            self.rendered_text = input_font.render(self.text, True, TEXT_COLOR)
            self.text_rect = self.rendered_text.get_rect(midleft=(self.rect.x + 10, self.rect.y + self.rect.height//2))
        except ValueError:
            # If conversion fails, reset to default
            self.text = self.default_value
            self.rendered_text = input_font.render(self.text, True, TEXT_COLOR)
            self.text_rect = self.rendered_text.get_rect(midleft=(self.rect.x + 10, self.rect.y + self.rect.height//2))

    def update(self):
        # Blink cursor every 500ms
        self.cursor_timer += 1
        if self.cursor_timer >= 30:  # 30 frames at 60fps ≈ 500ms
            self.cursor_timer = 0
            self.cursor_visible = not self.cursor_visible

    def draw(self, screen):
        # Draw the input field background
        pygame.draw.rect(screen, self.color, self.rect, border_radius=5)
        pygame.draw.rect(screen, BORDER_COLOR, self.rect, 1, border_radius=5)
        
        # Render the text
        if self.text:
            screen.blit(self.rendered_text, self.text_rect)
        
        # Draw cursor if active
        if self.active and self.cursor_visible:
            cursor_pos = self.rect.x + 10 + self.rendered_text.get_width() + 2
            pygame.draw.line(screen, TEXT_COLOR, 
                             (cursor_pos, self.rect.y + 8), 
                             (cursor_pos, self.rect.y + self.rect.height - 8), 2)

    def get_value(self):
        if not self.text:
            return self.default_value
        
        try:
            if self.decimal:
                value = float(self.text)
                # Apply min/max constraints on validation
                if self.min_value is not None and value < self.min_value:
                    return self.min_value
                elif self.max_value is not None and value > self.max_value:
                    return self.max_value
                return value
            else:
                value = int(self.text)
                # Apply min/max constraints on validation
                if self.min_value is not None and value < self.min_value:
                    return self.min_value
                elif self.max_value is not None and value > self.max_value:
                    return self.max_value
                return value
        except ValueError:
            return self.default_value

# Button class
class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = BUTTON_COLOR
        self.text = text
        self.rendered_text = button_font.render(text, True, TEXT_COLOR)
        self.text_rect = self.rendered_text.get_rect(center=self.rect.center)
        self.hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
            self.color = BUTTON_HOVER_COLOR if self.hovered else BUTTON_COLOR
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, screen):
        # Draw button with 3D effect
        pygame.draw.rect(screen, self.color, self.rect, border_radius=5)
        
        # Add highlight for 3D effect
        highlight_rect = pygame.Rect(self.rect.left + 1, self.rect.top + 1, 
                                   self.rect.width - 2, 3)
        pygame.draw.rect(screen, (min(self.color[0] + 30, 255), min(self.color[1] + 30, 255), min(self.color[2] + 30, 255)),
                        highlight_rect, border_radius=2)
        
        # Add shadow for 3D effect
        shadow_rect = pygame.Rect(self.rect.left + 1, self.rect.bottom - 3, 
                                self.rect.width - 2, 2)
        pygame.draw.rect(screen, (max(self.color[0] - 30, 0), max(self.color[1] - 30, 0), max(self.color[2] - 30, 0)), 
                        shadow_rect, border_radius=2)
        
        # Draw button text
        screen.blit(self.rendered_text, self.text_rect)

def run_vis_py(population_size, generations, mutation_rate, n):
    """Run vis.py with the specified parameters"""
    # Create a temporary file to modify vis.py parameters
    temp_script = """
import importlib
import vis

# Define a modified main function that uses our parameters
def modified_main():
    # Set the global variables explicitly
    vis.POPULATION_SIZE = {pop_size}
    vis.GENERATIONS = {gens}
    vis.MUTATION_RATE = {mut_rate}
    
    # Override N directly
    N = {n_val}
    
    # --- Visualization Setup ---
    # Define a wrapper function that calls run_experiment with the callback
    def ga_runner():
        print(f"Starting GA for N={{N}}...")
        # Capture the final ID and the lookup table
        generation, found, final_best_id, lookup_table = vis.run_experiment(
            N, vis.POPULATION_SIZE, vis.GENERATIONS, vis.MUTATION_RATE,
            callback=vis.update_visualization_data
        )
        print(f"GA finished after {{generation}} generations. Solution found: {{found}}")
        print(f"Final best individual ID: {{final_best_id}}")
        print(f"Total individuals created: {{len(lookup_table)}}")
        
        # Print detailed information about the final solution
        if final_best_id and final_best_id in lookup_table:
            final_solution = lookup_table[final_best_id]
            fitness = vis.calculate_fitness(final_solution, N)
            
            print("\\n--- FINAL SOLUTION DETAILS ---")
            print(f"Chromosome: {{final_solution['chromosome']}}")
            print(f"Fitness: {{fitness}}")
            print(f"Generation: {{final_solution['generation']}}")
            
            # Print parent information if available
            parents = final_solution.get('parents')
            if parents:
                p1_id = parents.get('parent1_id')
                p2_id = parents.get('parent2_id')
                
                if p1_id and p1_id in lookup_table:
                    p1 = lookup_table[p1_id]
                    print(f"\\nParent 1 (ID: {{p1_id}}):")
                    print(f"Chromosome: {{p1['chromosome']}}")
                    print(f"Fitness: {{vis.calculate_fitness(p1, N)}}")
                    print(f"Generation: {{p1['generation']}}")
                
                if p2_id and p2_id in lookup_table:
                    p2 = lookup_table[p2_id]
                    print(f"\\nParent 2 (ID: {{p2_id}}):")
                    print(f"Chromosome: {{p2['chromosome']}}")
                    print(f"Fitness: {{vis.calculate_fitness(p2, N)}}")
                    print(f"Generation: {{p2['generation']}}")
            
            # Print crossover information
            crossover_point = final_solution.get('crossover_point')
            if crossover_point is not None:
                print(f"\\nCrossover point: {{crossover_point}}")
                print("Crossover pattern: ", end="")
                for i in range(N):
                    if i < crossover_point:
                        print("1", end="")
                    else:
                        print("2", end="")
                print(" (1=from parent1, 2=from parent2)")
            
            # Print mutation information
            mutation_info = final_solution.get('mutation')
            if mutation_info:
                print(f"\\nMutation applied: {{mutation_info.get('applied', False)}}")
                if mutation_info.get('applied'):
                    positions = mutation_info.get('positions')
                    if positions:
                        print(f"Mutation positions: {{positions}}")
            
            print("\\n--- GENE ORIGIN ANALYSIS ---")
            if parents and p1_id in lookup_table and p2_id in lookup_table:
                p1 = lookup_table[p1_id]
                p2 = lookup_table[p2_id]
                
                print("Position: ", end="")
                for i in range(N):
                    print(f"{{i:2d}}", end=" ")
                print()
                
                print("Child:    ", end="")
                for gene in final_solution['chromosome']:
                    print(f"{{gene:2d}}", end=" ")
                print()
                
                print("Parent 1: ", end="")
                for gene in p1['chromosome']:
                    print(f"{{gene:2d}}", end=" ")
                print()
                
                print("Parent 2: ", end="")
                for gene in p2['chromosome']:
                    print(f"{{gene:2d}}", end=" ")
                print()
                
                # Get the recorded crossover point
                crossover_point = final_solution.get('crossover_point')
                if crossover_point is None:
                    crossover_point = N // 2  # Default fallback
                
                # Get mutation information
                mutation_info = final_solution.get('mutation', {{}})
                mutation_positions = mutation_info.get('positions', ()) if mutation_info.get('applied', False) else ()
                
                print("Origin:   ", end="")
                for i in range(N):
                    child_gene = final_solution['chromosome'][i]
                    
                    # First check if this position was mutated
                    if mutation_positions and i in mutation_positions:
                        print(" M", end=" ")  # Mutation
                    # For positions before crossover point, should come from parent 1
                    elif i < crossover_point:
                        if child_gene == p1['chromosome'][i]:
                            print(" 1", end=" ")  # From parent 1 in same position
                        else:
                            print(" ?", end=" ")  # Unexpected value
                    # For positions after crossover point
                    else:
                        # The gene could be from any position in parent 2 (not necessarily the same index)
                        # due to permutation constraint handling during crossover
                        if child_gene in p2['chromosome']:
                            print(" 2", end=" ")  # From parent 2 (some position)
                        else:
                            print(" ?", end=" ")  # Unexpected value
                print(" (1=from parent1, 2=from parent2, M=mutation, ?=unexpected)")
                
                # Add explanation of the permutation crossover
                print("\\nNote: For a permutation problem like N-Queens, crossover works differently than in standard GAs.")
                print("The algorithm:")
                print(f"1. Takes genes from Parent 1 up to position {{crossover_point-1}}")
                print(f"2. Takes remaining genes from Parent 2, but skips values already used")
                print("3. This ensures no duplicate values occur in the child chromosome")
                print("4. This means genes after the crossover point come from Parent 2 but may be from different positions")
                print("   in Parent 2's chromosome than their positions in the child chromosome.")
        
        # Additional analysis to match visualization
        print("\\nThis detailed output can be compared with the graphical visualization to verify")
        print("the crossover points and mutation indicators shown in the FILHO board.")

    # Run the visualization, passing the GA runner function
    vis.run_visualization(ga_runner)

# Run our modified main function
modified_main()
""".format(
        pop_size=population_size,
        gens=generations,
        mut_rate=mutation_rate,
        n_val=n
    )
    
    # Write to a temporary file
    with open("temp_executor.py", "w") as f:
        f.write(temp_script)
    
    try:
        # Run the script as a subprocess
        process = subprocess.Popen([sys.executable, "temp_executor.py"])
        
        # Return the process in case we need to handle it later
        return process
    except Exception as e:
        print(f"Error running vis.py: {e}")
        return None

def main():
    # Create input fields
    population_field = InputField(250, 100, 150, 40, 30, numeric_only=True, min_value=2, max_value=1000)
    generations_field = InputField(250, 160, 150, 40, 150, numeric_only=True, min_value=1, max_value=10000)
    mutation_rate_field = InputField(250, 220, 150, 40, 0.1, numeric_only=True, min_value=0.0, max_value=1.0, decimal=True)
    n_field = InputField(250, 280, 150, 40, 6, numeric_only=True, min_value=4, max_value=15)
    
    # Create run button
    run_button = Button(WIDTH//2 - 75, 350, 150, 50, "Run")
    
    clock = pygame.time.Clock()
    running = True
    status_message = ""
    status_color = TEXT_COLOR
    active_process = None
    
    try:
        # Main loop
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Handle input fields
                population_field.handle_event(event)
                generations_field.handle_event(event)
                mutation_rate_field.handle_event(event)
                n_field.handle_event(event)
                
                # Handle run button (only if no process is currently running)
                if active_process is None and run_button.handle_event(event):
                    try:
                        pop_size = population_field.get_value()
                        generations = generations_field.get_value()
                        mutation_rate = mutation_rate_field.get_value()
                        n_val = n_field.get_value()
                        
                        print("Running N-Queens GA with:")
                        print(f"  Population Size: {pop_size}")
                        print(f"  Generations: {generations}")
                        print(f"  Mutation Rate: {mutation_rate}")
                        print(f"  N: {n_val}")
                        
                        # Run vis.py with these parameters
                        active_process = run_vis_py(pop_size, generations, mutation_rate, n_val)
                        
                        if active_process:
                            status_message = "Running N-Queens Genetic Algorithm..."
                            status_color = (50, 200, 50)  # Green
                        else:
                            status_message = "Failed to start the process."
                            status_color = (200, 50, 50)  # Red
                            
                    except Exception as e:
                        status_message = f"Error: {str(e)}"
                        status_color = (200, 50, 50)  # Red
                        print(f"Error: {e}")
            
            # Check if the process is still running
            if active_process:
                if active_process.poll() is not None:
                    # Process has finished
                    if active_process.returncode == 0:
                        status_message = "Algorithm completed successfully!"
                    else:
                        status_message = f"Process ended with code {active_process.returncode}"
                        status_color = (200, 50, 50)  # Red
                    active_process = None  # Reset for next run
            
            # Update
            population_field.update()
            generations_field.update()
            mutation_rate_field.update()
            n_field.update()
            
            # Draw
            screen.fill(BACKGROUND_COLOR)
            
            # Draw title
            title_surface = title_font.render("N-Queens Genetic Algorithm Executor", True, TITLE_COLOR)
            title_rect = title_surface.get_rect(center=(WIDTH//2, 40))
            screen.blit(title_surface, title_rect)
            
            # Draw labels and input fields
            pop_label = label_font.render("Population Size:", True, TEXT_COLOR)
            screen.blit(pop_label, (50, 110))
            population_field.draw(screen)
            
            gen_label = label_font.render("Generations:", True, TEXT_COLOR)
            screen.blit(gen_label, (50, 170))
            generations_field.draw(screen)
            
            mut_label = label_font.render("Mutation Rate:", True, TEXT_COLOR)
            screen.blit(mut_label, (50, 230))
            mutation_rate_field.draw(screen)
            
            n_label = label_font.render("N (board size):", True, TEXT_COLOR)
            screen.blit(n_label, (50, 290))
            n_field.draw(screen)
            
            # Draw run button (different color if process is running)
            if active_process is None:
                run_button.draw(screen)
            else:
                # Draw a disabled run button
                disabled_rect = pygame.Rect(run_button.rect)
                pygame.draw.rect(screen, (100, 100, 100), disabled_rect, border_radius=5)
                disabled_text = button_font.render("Running...", True, (180, 180, 180))
                disabled_rect = disabled_text.get_rect(center=run_button.rect.center)
                screen.blit(disabled_text, disabled_rect)
            
            # Draw status message
            if status_message:
                status_surface = label_font.render(status_message, True, status_color)
                status_rect = status_surface.get_rect(center=(WIDTH//2, 410))
                screen.blit(status_surface, status_rect)
            
            pygame.display.flip()
            clock.tick(60)
    
    finally:
        # Cleanup - runs regardless of how the program exits
        pygame.quit()
        if os.path.exists("temp_executor.py"):
            try:
                os.remove("temp_executor.py")
                print("Temporary file removed.")
            except Exception as e:
                print(f"Error removing temporary file: {e}")
        sys.exit()

if __name__ == "__main__":
    main() 
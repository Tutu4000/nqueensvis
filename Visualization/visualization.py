import pygame
import sys
import math
import threading
import time
from typing import Dict, Any, Optional, Tuple, List # Added List
import random

# Constants
WIDTH, HEIGHT = 950, 550 # Increased height for mode switch buttons etc.
BOARD_AREA_WIDTH = 300
BOARD_SIZE = 250
PADDING = 20
INFO_HEIGHT = 50
BUTTON_HEIGHT = 30
BUTTON_WIDTH = 100
MODE_BUTTON_WIDTH = 150

# Modern color scheme
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BACKGROUND_COLOR = (40, 40, 40)  # Dark background
LIGHT_SQUARE = (232, 235, 239)  # Light cream/white
DARK_SQUARE = (125, 150, 105)   # Green shade for dark squares
QUEEN_COLOR = (255, 0, 0)       # Bright red for queens
TREE_NODE_COLOR = (100, 100, 200)
TREE_LINE_COLOR = (180, 180, 180)  # Lighter lines for better visibility
BEST_NODE_COLOR = (50, 180, 50)
BUTTON_COLOR = (80, 120, 130)   # Teal-like color for buttons
BUTTON_TEXT_COLOR = WHITE
TITLE_COLOR = WHITE
INFO_TEXT_COLOR = WHITE
GENERATION_TEXT_COLOR = WHITE

# Crossover and mutation indicators
P1_COLOR = (70, 140, 255)    # Blue for genes from parent 1
P2_COLOR = (255, 100, 100)   # Red for genes from parent 2
MUTATION_COLOR = (255, 240, 0)  # Yellow for mutations
CROSSOVER_MARKER_COLOR = (100, 255, 100)  # Green marker for crossover points

# Visualization Modes
VIEW_MODE_LINEAGE = "Lineage"
VIEW_MODE_TREE = "Tree"

# --- Shared Data (Updated by GA thread via callback) ---
n_queens = 8
_best_solution_internal: Optional[Dict[str, Any]] = None # Stores the *data* of the current best
_algorithm_finished_internal = False
_final_best_solution_id: Optional[Tuple[int, int]] = None # ID of the final best solution
_all_individuals_lookup_internal: Dict[Tuple[int, int], Dict[str, Any]] = {} # Lookup for all individuals
ga_lock = threading.Lock()
# --- End Shared Data ---

# --- UI State ---
view_stack: List[Dict[str, Any]] = [] # Stack for Lineage view (stores individual data)
current_view_mode = VIEW_MODE_LINEAGE # Start with lineage view
# Tree View State
tree_zoom = 1.0
tree_offset_x = 0
tree_offset_y = 0
dragging = False
drag_start_pos = (0, 0)
drag_start_offset = (0, 0)
# --- End UI State ---

# --- Helper for Inverse Transform (screen -> world) ---
# Defined at module level for access by run_visualization
def screen_to_world(screen_pos, offset, current_zoom, screen_center):
    """Converts screen coordinates to world coordinates."""
    center_x, center_y = screen_center
    if current_zoom == 0: return (0, 0) # Avoid division by zero
    offset_x, offset_y = offset
    temp_x = screen_pos[0] - offset_x
    temp_y = screen_pos[1] - offset_y
    world_x = center_x + (temp_x - center_x) / current_zoom
    world_y = center_y + (temp_y - center_y) / current_zoom
    return world_x, world_y

# Attempt to import calculate_fitness from vis.py
try:
    from vis import calculate_fitness as imported_calculate_fitness
except ImportError:
    print("Warning: Could not import 'calculate_fitness' from vis.py. Using fallback definition.")
    # Fallback definition if running standalone or import issues
    def imported_calculate_fitness(solution, N):
        if solution is None or "chromosome" not in solution: return None
        chromosome = solution["chromosome"]
        conflicts = 0
        for i in range(N):
            for j in range(i + 1, N):
                if i == j: continue # Should not happen with range(i+1, N)
                # Check columns (redundant check with permutation encoding, but safe)
                if chromosome[i] == chromosome[j]:
                    conflicts += 1
                # Check diagonals
                if abs(chromosome[i] - chromosome[j]) == abs(i - j):
                    conflicts += 1
        # Fitness = 1 for 0 conflicts, decreases exponentially otherwise
        try:
            return math.exp(-conflicts)
        except OverflowError:
            return 0.0 # Handle potential overflow for large conflict counts

def calculate_cell_size(n):
    return BOARD_SIZE // n if n > 0 else BOARD_SIZE # Avoid division by zero

def draw_board(screen, chromosome_data, n, position, title, fitness):
    cell_size = calculate_cell_size(n)
    current_board_size = cell_size * n
    board_rect = pygame.Rect(position[0], position[1] + INFO_HEIGHT, current_board_size, current_board_size)

    # Draw title in bold style
    font_title = pygame.font.Font(None, 28)  # Slightly larger font
    
    # Simplify the title display to match the image
    if title in ["Pai", "Mae", "Filho"]:
        if title == "Pai":
            display_title = "PAI 1"
        elif title == "Mae":
            display_title = "PAI 2"
        elif title == "Filho":
            display_title = "FILHO"
        title_surf = font_title.render(display_title, True, TITLE_COLOR)
    else:
        fitness_str = f"{fitness:.3f}" if fitness is not None else "N/A"
        gen_str = f" (Gen: {chromosome_data.get('generation', '?')})" if isinstance(chromosome_data, dict) else ""
        id_str = f" ID: {chromosome_data.get('id', '?')}" if isinstance(chromosome_data, dict) else ""
        title_text = f"{title} Fitness: {fitness_str}{gen_str}{id_str}"
        title_surf = font_title.render(title_text, True, TITLE_COLOR)
    
    title_rect = title_surf.get_rect(center=(board_rect.centerx, position[1] + INFO_HEIGHT // 2))
    screen.blit(title_surf, title_rect)

    # Draw board outline with rounded corners
    pygame.draw.rect(screen, BLACK, board_rect, 1, border_radius=5)

    actual_chromosome = None
    if isinstance(chromosome_data, dict):
        actual_chromosome = chromosome_data.get('chromosome')

    if actual_chromosome is None or n <= 0:
        pygame.draw.rect(screen, LIGHT_SQUARE, board_rect.inflate(-2, -2), border_radius=5)
        return board_rect

    for r in range(n):
        for c in range(n):
            color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (board_rect.left + c * cell_size, board_rect.top + r * cell_size, cell_size, cell_size))

    # Draw queens as filled circles with a subtle gradient effect
    queen_radius = max(1, cell_size // 2 - 4)
    for col, row in enumerate(actual_chromosome):
        if 0 <= row < n:
            center_x = board_rect.left + col * cell_size + cell_size // 2
            center_y = board_rect.top + row * cell_size + cell_size // 2
            
            # Draw main queen circle
            pygame.draw.circle(screen, QUEEN_COLOR, (center_x, center_y), queen_radius)
            
            # Add a small white highlight to give a 3D effect
            highlight_radius = max(1, queen_radius // 3)
            highlight_offset = max(1, queen_radius // 4)
            pygame.draw.circle(screen, (255, 100, 100), 
                             (center_x - highlight_offset, center_y - highlight_offset), 
                             highlight_radius)

    return board_rect

def draw_button(screen, text, position, size, enabled=True):
    button_rect = pygame.Rect(position, size)
    bg_color = BUTTON_COLOR if enabled else (80, 80, 80)  # Darker gray when disabled
    text_color = BUTTON_TEXT_COLOR if enabled else (150, 150, 150)  # Lighter gray text when disabled
    
    # Draw button with rounded corners and no border
    pygame.draw.rect(screen, bg_color, button_rect, border_radius=5)
    
    # Draw subtle highlight on top edge for 3D effect
    highlight_rect = pygame.Rect(button_rect.left + 1, button_rect.top + 1, 
                               button_rect.width - 2, 3)
    pygame.draw.rect(screen, (min(bg_color[0] + 30, 255), min(bg_color[1] + 30, 255), min(bg_color[2] + 30, 255)), 
                    highlight_rect, border_radius=2)
    
    # Draw subtle shadow on bottom edge
    shadow_rect = pygame.Rect(button_rect.left + 1, button_rect.bottom - 3, 
                            button_rect.width - 2, 2)
    pygame.draw.rect(screen, (max(bg_color[0] - 30, 0), max(bg_color[1] - 30, 0), max(bg_color[2] - 30, 0)), 
                    shadow_rect, border_radius=2)
    
    font = pygame.font.Font(None, 20)
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=button_rect.center)
    screen.blit(text_surf, text_rect)
    return button_rect

# --- Callback Function ---
def update_visualization_data(best_solution, n, finished=False, all_individuals=None):
    # Accepts the full lookup table and best ID when finished
    global _best_solution_internal, _algorithm_finished_internal, n_queens, view_stack
    global _final_best_solution_id, _all_individuals_lookup_internal
    with ga_lock:
        n_queens = n
        _best_solution_internal = best_solution.copy() if best_solution else None
        _algorithm_finished_internal = finished
        if finished:
            _all_individuals_lookup_internal = all_individuals if all_individuals else {}
            _final_best_solution_id = _best_solution_internal.get("id") if _best_solution_internal else None
            # Initialize view stack only if we have a final solution
            if _best_solution_internal:
                view_stack = [_best_solution_internal]
            else:
                view_stack = [] # Clear stack if no solution found
# --- End Callback ---

# --- Tree Drawing Functions ---
# Base size for mini-boards in the tree
MINI_BOARD_BASE_SIZE = 60 # Increased base size
MINI_BOARD_MIN_ZOOM_SIZE = 15 # Increased min size
MINI_BOARD_MAX_ZOOM_SIZE = 240 # Increased max size significantly
TREE_VERTICAL_SPACING = 80 # Increased spacing slightly more

def draw_mini_board(screen, chromosome, n, center_pos, size, node_data=None, lookup_table=None):
    """Draw a mini board for tree view, with optional crossover/mutation visualization if node_data is provided"""
    if not chromosome or n <= 0 or size <= 0:
        return # Don't draw if invalid

    cell_size = size / n
    if cell_size < 1: return # Don't draw if cells are too small

    board_rect = pygame.Rect(0, 0, size, size)
    board_rect.center = center_pos

    # Draw simple border
    pygame.draw.rect(screen, BLACK, board_rect, 1, border_radius=3)

    # Draw squares
    for r in range(n):
        for c in range(n):
            color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (board_rect.left + c * cell_size, board_rect.top + r * cell_size, math.ceil(cell_size), math.ceil(cell_size)))

    # Determine if we can show crossover/mutation information
    parent_map = []
    mutation_indices = []
    crossover_points = []
    
    if node_data and lookup_table and node_data.get('parents'):
        parents = node_data.get('parents', {})
        # Make sure parents is not None and has the required keys
        if parents is not None:
            p1_id = parents.get('parent1_id')
            p2_id = parents.get('parent2_id')
            
            # Only process if we have both parents
            if p1_id and p2_id:
                p1_data = lookup_table.get(p1_id)
                p2_data = lookup_table.get(p2_id)
                if p1_data and p2_data:
                    parent_map, mutation_indices, crossover_points = get_crossover_info(node_data, p1_data, p2_data)
    
    # Draw crossover indicators (if applicable and board is big enough)
    if crossover_points and size >= 40:
        for crossover_point in crossover_points:
            x_pos = board_rect.left + crossover_point * cell_size
            # Draw a thinner vertical line for crossover points
            pygame.draw.line(screen, CROSSOVER_MARKER_COLOR, 
                           (x_pos, board_rect.top), 
                           (x_pos, board_rect.bottom), 
                           max(1, int(size / 60)))

    # Draw queens (simple circles with a highlight effect)
    queen_radius = max(1, int(cell_size / 2 * 0.8)) # Smaller queens
    for col, row in enumerate(chromosome):
        if 0 <= row < n:
            center_x = board_rect.left + col * cell_size + cell_size / 2
            center_y = board_rect.top + row * cell_size + cell_size / 2
            
            # Choose color based on parent source (if available)
            queen_color = QUEEN_COLOR
            if col < len(parent_map):
                if parent_map[col] == 1:
                    queen_color = P1_COLOR
                elif parent_map[col] == 2:
                    queen_color = P2_COLOR
                elif parent_map[col] == 0:
                    queen_color = MUTATION_COLOR
            
            # Draw main queen circle
            pygame.draw.circle(screen, queen_color, (int(center_x), int(center_y)), queen_radius)
            
            # Add small highlight if queens are big enough
            if queen_radius > 3:
                highlight_radius = max(1, queen_radius // 3)
                highlight_offset = max(1, queen_radius // 4)
                
                # Determine highlight color
                if queen_color == P1_COLOR:
                    highlight_color = (150, 200, 255)  # Lighter blue
                elif queen_color == P2_COLOR:
                    highlight_color = (255, 180, 180)  # Lighter red
                elif queen_color == MUTATION_COLOR:
                    highlight_color = (255, 255, 150)  # Lighter yellow
                else:
                    highlight_color = (255, 180, 180)  # Default
                
                pygame.draw.circle(screen, highlight_color, 
                                (int(center_x - highlight_offset), int(center_y - highlight_offset)), 
                                highlight_radius)
                
            # For mutations in larger boards, add an indicator
            if col in mutation_indices and size >= 40:
                star_radius = queen_radius + 2
                pygame.draw.circle(screen, MUTATION_COLOR, (int(center_x), int(center_y)), star_radius, 1)

def get_all_ancestors(node_id, lookup_table) -> set:
    """Recursively finds all unique ancestor IDs for a given node ID."""
    ancestors = set()
    queue = [node_id]
    processed = set()

    while queue:
        current_id = queue.pop(0)
        if not current_id or current_id in processed:
            continue
        processed.add(current_id)
        ancestors.add(current_id)

        node_data = lookup_table.get(current_id)
        if node_data and node_data.get('parents'):
            p1_id = node_data['parents'].get('parent1_id')
            p2_id = node_data['parents'].get('parent2_id')
            if p1_id and p1_id not in processed:
                queue.append(p1_id)
            if p2_id and p2_id not in processed:
                queue.append(p2_id)
    return ancestors

def get_node_position(node_id, layout_cache):
    # Placeholder: needs actual layout algorithm
    # For now, just return a position based on generation and index if cached
    if node_id in layout_cache:
        return layout_cache[node_id]
    else:
        # Fallback: Position based on ID (will look messy)
        gen, idx = node_id
        return (50 + idx * 20, HEIGHT - 50 - gen * 40)

def calculate_tree_layout(root_id, lookup_table):
    if not root_id or root_id not in lookup_table:
        return {}

    # 1. Find all ancestors of the root node
    ancestor_ids = get_all_ancestors(root_id, lookup_table)

    # 2. Group ancestors by generation
    layout = {}
    nodes_at_gen = {} # {gen: [node_id, ...]}
    max_gen = 0
    for node_id in ancestor_ids:
        node_data = lookup_table.get(node_id)
        if node_data:
            gen = node_data.get('generation', 0)
            if gen not in nodes_at_gen:
                nodes_at_gen[gen] = []
            nodes_at_gen[gen].append(node_id)
            max_gen = max(max_gen, gen)

    # 3. Assign coordinates (using per-generation centering)
    total_height = (max_gen + 1) * TREE_VERTICAL_SPACING
    start_y = PADDING + INFO_HEIGHT # Start drawing below buttons

    for gen, nodes in sorted(nodes_at_gen.items()):
        y = start_y + gen * TREE_VERTICAL_SPACING
        num_nodes = len(nodes)
        
        # Sort nodes within a generation by index for consistent X placement
        nodes.sort(key=lambda nid: nid[1] if isinstance(nid, tuple) and len(nid) > 1 else 0)
        
        # Calculate horizontal spacing factor *for this specific generation*
        width_factor_gen = (WIDTH - 2 * PADDING) / max(1, num_nodes)

        for i, node_id in enumerate(nodes):
            # Distribute nodes horizontally using the generation-specific factor
            x = PADDING + (i + 0.5) * width_factor_gen
            layout[node_id] = (x, y)

    return layout

# --- Helper function to draw an arrow ---
def draw_arrow(screen, color, start_pos, end_pos, shorten_by=0, arrow_size=6, line_thickness=1):
    """Draws a line with an arrowhead, stopping short of the end_pos."""
    try:
        # Vector from start to end
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = math.hypot(dx, dy)

        if length <= shorten_by or length == 0: # If points too close or same, don't draw
            return

        # Normalize vector
        dx /= length
        dy /= length

        # Calculate new endpoint for the line segment
        line_end_x = start_pos[0] + dx * (length - shorten_by)
        line_end_y = start_pos[1] + dy * (length - shorten_by)
        line_end_pos = (line_end_x, line_end_y)

        # Draw the shortened line
        pygame.draw.line(screen, color, start_pos, line_end_pos, line_thickness)

        # Calculate angle and points for arrowhead at the *new* end point
        angle = math.atan2(start_pos[1] - line_end_y, start_pos[0] - line_end_x) # Angle towards original start
        
        # Arrowhead points relative to line_end_pos
        p1_x = line_end_x + arrow_size * math.cos(angle + math.pi / 6)
        p1_y = line_end_y + arrow_size * math.sin(angle + math.pi / 6)
        p2_x = line_end_x + arrow_size * math.cos(angle - math.pi / 6)
        p2_y = line_end_y + arrow_size * math.sin(angle - math.pi / 6)
        
        pygame.draw.polygon(screen, color, [(int(line_end_x), int(line_end_y)), (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y))])

    except (ValueError, OverflowError, ZeroDivisionError) as e:
        # Avoid crashing if calculation fails
        # print(f"Could not draw arrow segment: {e}")
        # Fallback: draw a simple line if shortening/arrow fails badly
        try:
            pygame.draw.line(screen, color, start_pos, end_pos, line_thickness)
        except Exception:
            pass # Give up if even basic line fails

def draw_tree_view(screen, root_id, lookup_table, layout_cache, camera_offset, zoom, n):
    if not root_id or not lookup_table or not layout_cache:
        font = pygame.font.Font(None, 36)
        text_surf = font.render("No solution/data for tree view", True, INFO_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(text_surf, text_rect)
        return

    # Get the set of ancestors to draw
    ancestor_ids = set(layout_cache.keys()) # Layout cache now only contains ancestors
    lines_to_draw = []

    # Build line list - only connect nodes within the ancestor set
    for node_id in ancestor_ids:
        node_data = lookup_table.get(node_id)
        if node_data and node_data.get('parents'):
            p1_id = node_data['parents'].get('parent1_id')
            p2_id = node_data['parents'].get('parent2_id')
            # Only add line if parent is also an ancestor
            if p1_id and p1_id in ancestor_ids:
                lines_to_draw.append((node_id, p1_id))
            if p2_id and p2_id in ancestor_ids:
                lines_to_draw.append((node_id, p2_id))

    # --- Apply Camera Transform ---
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    camera_offset_x, camera_offset_y = camera_offset # Unpack for clarity
    def transform(pos):
        try:
            # Check if pos is a valid sequence of numbers
            if not (isinstance(pos, (list, tuple)) and len(pos) == 2 and isinstance(pos[0], (int, float)) and isinstance(pos[1], (int, float))):
                 print(f"Warning: Invalid position value in layout cache: {pos}")
                 return (-1000, -1000) # Default off-screen

            # Apply zoom centered around screen center
            scaled_x = center_x + (pos[0] - center_x) * zoom
            scaled_y = center_y + (pos[1] - center_y) * zoom

            # Add offset
            final_x = scaled_x + camera_offset_x
            final_y = scaled_y + camera_offset_y

            # Ensure the result is finite before converting to int
            if not math.isfinite(final_x) or not math.isfinite(final_y):
                # Return a default off-screen position if calculation resulted in inf/nan
                # print(f"Warning: Non-finite coordinate generated: ({final_x}, {final_y})")
                return (-1000, -1000)
            
            # Return the integer tuple for drawing
            return int(final_x), int(final_y)
        except (TypeError, IndexError, ValueError, OverflowError) as e:
            # Handle potential errors if pos is malformed or math fails
            print(f"Error during transform calculation: {e}, Pos: {pos}")
            # Return a default off-screen position
            return (-1000, -1000)

    # --- Draw Lines as Arrows --- (only for ancestors)
    # Calculate board size based on current zoom, needed for shortening lines and drawing nodes
    scale_exponent = 1.1
    scaled_base_size = MINI_BOARD_BASE_SIZE * (zoom ** scale_exponent)
    current_mini_board_size = int(max(MINI_BOARD_MIN_ZOOM_SIZE, min(MINI_BOARD_MAX_ZOOM_SIZE, scaled_base_size)))
    shorten_amount = current_mini_board_size // 2 + 2 # Shorten by half board size + gap

    for child_id, parent_id in lines_to_draw:
        if child_id in layout_cache and parent_id in layout_cache:
            child_pos = transform(layout_cache[child_id])
            parent_pos = transform(layout_cache[parent_id])
            if child_pos == parent_pos: continue

            # Pass shorten_amount to draw_arrow
            draw_arrow(screen, TREE_LINE_COLOR, parent_pos, child_pos,
                       shorten_by=shorten_amount,
                       arrow_size=max(3, int(6 * zoom)),
                       line_thickness=max(1, int(2 * zoom)))  # Make lines more visible

    # --- Draw Nodes as Mini Boards --- (only for ancestors)
    mini_board_size = current_mini_board_size # USE the size calculated above
    screen_rect = screen.get_rect()
    info_font = pygame.font.Font(None, 14) # Small font for info

    for node_id in ancestor_ids:
        if node_id in layout_cache:
            node_data = lookup_table.get(node_id)
            if not node_data: continue
            pos = transform(layout_cache[node_id])
            margin = mini_board_size
            # Culling
            if pos[0] < -margin or pos[0] > WIDTH + margin or pos[1] < -margin or pos[1] > HEIGHT + margin:
                continue
                
            # Ensure we have a valid chromosome before drawing
            chromosome = node_data.get('chromosome')
            if not chromosome or not isinstance(chromosome, (list, tuple)) or len(chromosome) != n:
                # Skip drawing this node if chromosome is invalid
                print(f"Warning: Invalid chromosome for node {node_id}")
                continue
            
            # Draw the board with crossover/mutation visualization
            draw_mini_board(screen, chromosome, n, pos, mini_board_size, node_data, lookup_table)

            # Draw Info Text Above Board
            fitness = imported_calculate_fitness(node_data, n)
            gen = node_data.get('generation', '?')
            fitness_str = f"{fitness:.2f}" if fitness is not None else "N/A"
            info_text = f"ID:{node_id} G:{gen} F:{fitness_str}"
            text_surf = info_font.render(info_text, True, INFO_TEXT_COLOR)
            text_rect = text_surf.get_rect()
            text_rect.centerx = pos[0]
            text_rect.bottom = pos[1] - mini_board_size // 2 - 2
            screen.blit(text_surf, text_rect)

            # Highlight root node
            if node_id == root_id:
                highlight_rect = pygame.Rect(0, 0, mini_board_size + 6, mini_board_size + 6)
                highlight_rect.center = pos
                pygame.draw.rect(screen, BEST_NODE_COLOR, highlight_rect, 2, border_radius=3)

# --- End Tree Drawing Functions ---

def run_visualization(ga_runner_func):
    global view_stack, current_view_mode # Allow modification
    global tree_zoom, tree_offset_x, tree_offset_y, dragging, drag_start_pos, drag_start_offset

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("N-Queens Genetic Algorithm Visualization")
    clock = pygame.time.Clock()

    ga_thread = threading.Thread(target=ga_runner_func, daemon=True)
    ga_thread.start()

    running = True
    # Button rects for lineage view
    p1_button_rect = None
    p2_button_rect = None
    back_button_rect = None
    # Button rects for mode switching
    lineage_mode_button_rect = None
    tree_mode_button_rect = None

    tree_layout_cache = {} # Cache calculated layout
    needs_layout_recalculation = True # Flag to recalculate layout when needed

    while running:
        # Copy shared data safely
        with ga_lock:
            current_best_during_run = _best_solution_internal
            local_n = n_queens
            local_finished = _algorithm_finished_internal
            local_lookup_table = _all_individuals_lookup_internal
            local_final_best_id = _final_best_solution_id

        # Recalculate tree layout if needed (only once after finished)
        if local_finished and needs_layout_recalculation and local_final_best_id:
            print("Calculating tree layout...")
            tree_layout_cache = calculate_tree_layout(local_final_best_id, local_lookup_table)
            needs_layout_recalculation = False
            print(f"Layout calculated for {len(tree_layout_cache)} nodes.")

        # --- Event Handling ---
        mouse_pos = pygame.mouse.get_pos()
        screen_center = (WIDTH // 2, HEIGHT // 2) # Use this tuple
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # --- Mode Switch Button Handling ---
            if local_finished and event.type == pygame.MOUSEBUTTONDOWN:
                if lineage_mode_button_rect and lineage_mode_button_rect.collidepoint(mouse_pos):
                    if current_view_mode != VIEW_MODE_LINEAGE:
                        current_view_mode = VIEW_MODE_LINEAGE
                        # Reset lineage view stack if switching back
                        if local_final_best_id and local_lookup_table.get(local_final_best_id):
                            view_stack = [local_lookup_table[local_final_best_id]]
                        else:
                            view_stack = []
                elif tree_mode_button_rect and tree_mode_button_rect.collidepoint(mouse_pos):
                    current_view_mode = VIEW_MODE_TREE

            # --- View-Specific Event Handling ---
            if local_finished and current_view_mode == VIEW_MODE_LINEAGE:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # (Lineage view button clicks - Pai, Mae, Voltar)
                    # This logic remains the same as before
                    if p1_button_rect and p1_button_rect.collidepoint(mouse_pos) and view_stack:
                         focused = view_stack[-1]
                         parents = focused.get('parents')
                         if parents:
                             p1_id = parents.get('parent1_id')
                             p1_data = local_lookup_table.get(p1_id)
                             if p1_data:
                                 if p1_data.get('parents'): # Check if grandparent exists
                                     view_stack.append(p1_data)
                                 else:
                                     print(f"Pai {p1_id} has no further lineage.")
                    elif p2_button_rect and p2_button_rect.collidepoint(mouse_pos) and view_stack:
                         focused = view_stack[-1]
                         parents = focused.get('parents')
                         if parents:
                             p2_id = parents.get('parent2_id')
                             p2_data = local_lookup_table.get(p2_id)
                             if p2_data:
                                 if p2_data.get('parents'):
                                     view_stack.append(p2_data)
                                 else:
                                     print(f"Mae {p2_id} has no further lineage.")
                    elif back_button_rect and back_button_rect.collidepoint(mouse_pos):
                         if len(view_stack) > 1:
                             view_stack.pop()

            elif local_finished and current_view_mode == VIEW_MODE_TREE:
                # --- Tree View Interactions (Zoom/Pan) ---
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click drag
                        dragging = True
                        drag_start_pos = mouse_pos
                        drag_start_offset = (tree_offset_x, tree_offset_y)
                    elif event.button == 4: # Scroll up (zoom in)
                        zoom_factor = 1.1
                        world_before = screen_to_world(mouse_pos, (tree_offset_x, tree_offset_y), tree_zoom, screen_center)
                        
                        # Apply zoom
                        tree_zoom *= zoom_factor
                        
                        # Calculate where the world point would land with new zoom & zero offset
                        projected_x = screen_center[0] + (world_before[0] - screen_center[0]) * tree_zoom
                        projected_y = screen_center[1] + (world_before[1] - screen_center[1]) * tree_zoom
                        
                        # Set the offset needed to bring that projected point to the mouse position
                        tree_offset_x = mouse_pos[0] - projected_x
                        tree_offset_y = mouse_pos[1] - projected_y
                    elif event.button == 5: # Scroll down (zoom out)
                        zoom_factor = 1 / 1.1
                        new_zoom = tree_zoom * zoom_factor
                        
                        # Prevent zooming out too much
                        if new_zoom < 0.1:
                             new_zoom = 0.1 # Clamp to minimum zoom
                             if tree_zoom <= 0.1: continue # Already at min zoom, do nothing
                             
                        world_before = screen_to_world(mouse_pos, (tree_offset_x, tree_offset_y), tree_zoom, screen_center)
                        
                        # Apply the new zoom
                        tree_zoom = new_zoom
                        
                        # Calculate where the world point would land with new zoom & zero offset
                        projected_x = screen_center[0] + (world_before[0] - screen_center[0]) * tree_zoom
                        projected_y = screen_center[1] + (world_before[1] - screen_center[1]) * tree_zoom
                        
                        # Set the offset needed to bring that projected point to the mouse position
                        tree_offset_x = mouse_pos[0] - projected_x
                        tree_offset_y = mouse_pos[1] - projected_y
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if dragging:
                        dx = mouse_pos[0] - drag_start_pos[0]
                        dy = mouse_pos[1] - drag_start_pos[1]
                        tree_offset_x = drag_start_offset[0] + dx
                        tree_offset_y = drag_start_offset[1] + dy

        # --- Drawing --- 
        screen.fill(BACKGROUND_COLOR)  # Use the dark background color

        # --- Draw Mode Switch Buttons (if finished) ---
        if local_finished:
            lineage_mode_button_rect = draw_button(screen, "Lineage View", (PADDING, PADDING), (MODE_BUTTON_WIDTH, BUTTON_HEIGHT), enabled=current_view_mode != VIEW_MODE_LINEAGE)
            tree_mode_button_rect = draw_button(screen, "Tree View", (PADDING + MODE_BUTTON_WIDTH + PADDING, PADDING), (MODE_BUTTON_WIDTH, BUTTON_HEIGHT), enabled=current_view_mode != VIEW_MODE_TREE)
            draw_y_offset = PADDING + BUTTON_HEIGHT + PADDING # Y position below buttons
        else:
            draw_y_offset = PADDING # Start drawing boards from top if not finished

        # --- Draw Content Based on Mode ---
        if not local_finished or current_view_mode == VIEW_MODE_LINEAGE:
            # --- Draw Lineage View ---
            focused_individual = None
            display_p1 = None
            display_p2 = None

            if not local_finished:
                 # Use the current best tracked during the run
                 focused_individual = current_best_during_run
                 if focused_individual:
                     parents = focused_individual.get('parents')
                     if parents:
                         p1_id = parents.get('parent1_id')
                         p2_id = parents.get('parent2_id')
                         # Need lookup table even during run now
                         display_p1 = local_lookup_table.get(p1_id)
                         display_p2 = local_lookup_table.get(p2_id)
            else: # Finished, Lineage Mode
                if view_stack:
                    focused_individual = view_stack[-1]
                    parents = focused_individual.get('parents')
                    if parents:
                         p1_id = parents.get('parent1_id')
                         p2_id = parents.get('parent2_id')
                         display_p1 = local_lookup_table.get(p1_id)
                         display_p2 = local_lookup_table.get(p2_id)

            cell_size = calculate_cell_size(local_n)
            current_board_size = cell_size * local_n if local_n > 0 else BOARD_SIZE
            board_y = draw_y_offset # Use offset Y

            total_board_width = 3 * current_board_size
            total_padding = WIDTH - total_board_width
            dynamic_padding = max(PADDING, total_padding // 4)

            p1_x = dynamic_padding
            p2_x = p1_x + current_board_size + dynamic_padding
            filho_x = p2_x + current_board_size + dynamic_padding

            p1_fitness = imported_calculate_fitness(display_p1, local_n) if display_p1 else None
            p2_fitness = imported_calculate_fitness(display_p2, local_n) if display_p2 else None
            filho_fitness = imported_calculate_fitness(focused_individual, local_n) if focused_individual else None

            # Draw generation title at the top
            if not local_finished:
                gen_number = 0
                if focused_individual and "generation" in focused_individual:
                    gen_number = focused_individual.get("generation", 0)
                
                # Draw generation title
                gen_font = pygame.font.Font(None, 36)
                gen_text = f"Geração: {gen_number}"
                gen_surf = gen_font.render(gen_text, True, GENERATION_TEXT_COLOR)
                gen_rect = gen_surf.get_rect(center=(WIDTH // 2, PADDING + 15))
                screen.blit(gen_surf, gen_rect)

            # Draw parent boards
            p1_board_rect = draw_board(screen, display_p1, local_n, (p1_x, board_y), "Pai", p1_fitness)
            p2_board_rect = draw_board(screen, display_p2, local_n, (p2_x, board_y), "Mae", p2_fitness)
            
            # Draw the child board with special crossover visualization 
            # Only when both parents are available
            if display_p1 and display_p2 and focused_individual:
                filho_board_rect = draw_filho_board(screen, focused_individual, display_p1, display_p2, 
                                                  local_n, (filho_x, board_y), "Filho", filho_fitness)
            else:
                # Fallback to regular board if we don't have both parents
                filho_board_rect = draw_board(screen, focused_individual, local_n, (filho_x, board_y), 
                                            "Filho", filho_fitness)

            button_y = max(p1_board_rect.bottom, p2_board_rect.bottom, filho_board_rect.bottom) + PADDING
            p1_button_rect = None
            p2_button_rect = None
            back_button_rect = None

            if local_finished:
                if display_p1 and display_p1.get('parents'):
                    button_p1_pos = (p1_board_rect.centerx - BUTTON_WIDTH // 2, button_y)
                    p1_button_rect = draw_button(screen, "Detalhes", button_p1_pos, (BUTTON_WIDTH, BUTTON_HEIGHT))
                if display_p2 and display_p2.get('parents'):
                    button_p2_pos = (p2_board_rect.centerx - BUTTON_WIDTH // 2, button_y)
                    p2_button_rect = draw_button(screen, "Detalhes", button_p2_pos, (BUTTON_WIDTH, BUTTON_HEIGHT))
                if len(view_stack) > 1:
                    button_back_pos = (filho_board_rect.centerx - BUTTON_WIDTH // 2, button_y)
                    back_button_rect = draw_button(screen, "Voltar", button_back_pos, (BUTTON_WIDTH, BUTTON_HEIGHT))

        elif local_finished and current_view_mode == VIEW_MODE_TREE:
            # --- Draw Tree View ---
            draw_tree_view(screen, local_final_best_id, local_lookup_table, tree_layout_cache, (tree_offset_x, tree_offset_y), tree_zoom, local_n)
            # Draw zoom/pan instructions
            font = pygame.font.Font(None, 18)
            text_surf = font.render("Scroll to Zoom, Drag to Pan", True, INFO_TEXT_COLOR)
            screen.blit(text_surf, (WIDTH - text_surf.get_width() - 5, HEIGHT - text_surf.get_height() - 5))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print("Visualization window closed.")

# --- Standalone Test Block ---
if __name__ == '__main__':
    print("Running visualization.py directly (for testing layout).")
    n_test = 5

    # Build a dummy lookup table and best ID
    test_lookup = {}
    id_counter = 0
    def add_node(gen, parents_list, lookup, chromosome=None):
        global id_counter
        node_id = (gen, id_counter)
        id_counter += 1
        parent_data = None
        if parents_list:
            parent_data = {"parent1_id": parents_list[0], "parent2_id": parents_list[1]}
        
        # Generate a chromosome if not provided
        if chromosome is None:
            chromosome = random.sample(range(n_test), n_test)
            
        node = {"id": node_id, "generation": gen, "chromosome": chromosome, "parents": parent_data}
        lookup[node_id] = node
        return node_id

    # Create initial population (gen 0)
    g0_1 = add_node(0, None, test_lookup)
    g0_2 = add_node(0, None, test_lookup)
    g0_3 = add_node(0, None, test_lookup)
    g0_4 = add_node(0, None, test_lookup)
    
    # Access the chromosomes for parents
    p1_chromosome = test_lookup[g0_1]["chromosome"]
    p2_chromosome = test_lookup[g0_2]["chromosome"]
    p3_chromosome = test_lookup[g0_3]["chromosome"]
    p4_chromosome = test_lookup[g0_4]["chromosome"]
    
    # Create gen 1 with simulated crossover and mutation
    # For g1_1: Take first half from p1, second half from p2
    crossover_point = n_test // 2
    child1_chromosome = p1_chromosome[:crossover_point] + p2_chromosome[crossover_point:]
    # Add a mutation
    mutation_index = random.randint(0, n_test-1)
    child1_chromosome[mutation_index] = (child1_chromosome[mutation_index] + 1) % n_test
    g1_1 = add_node(1, [g0_1, g0_2], test_lookup, child1_chromosome)
    
    # For g1_2: Take alternating genes from p3 and p4
    child2_chromosome = []
    for i in range(n_test):
        if i % 2 == 0:
            child2_chromosome.append(p3_chromosome[i])
        else:
            child2_chromosome.append(p4_chromosome[i])
    g1_2 = add_node(1, [g0_3, g0_4], test_lookup, child2_chromosome)
    
    # For final best solution: combine features from both gen 1 parents
    c1_chromosome = test_lookup[g1_1]["chromosome"]
    c2_chromosome = test_lookup[g1_2]["chromosome"]
    crossover_point = n_test // 3 * 2  # 2/3 through the chromosome
    best_chromosome = c1_chromosome[:crossover_point] + c2_chromosome[crossover_point:]
    # Add another mutation for demonstration
    mutation_index = n_test - 2  # Near the end
    best_chromosome[mutation_index] = (best_chromosome[mutation_index] + 2) % n_test
    
    g2_1 = add_node(2, [g1_1, g1_2], test_lookup, best_chromosome) # Final best

    test_final_best_id = g2_1

    # Simulate callback update for finished state
    update_visualization_data(test_lookup[test_final_best_id], n_test, finished=True, all_individuals=test_lookup)

    def dummy_runner():
        print("Dummy GA runner started and finished immediately for UI test.")

    run_visualization(dummy_runner)
    sys.exit()

# New function to determine crossover information
def get_crossover_info(child, p1, p2):
    """Determines for each gene in the child which parent it came from and identifies mutations.
    Returns a tuple of (parent_map, mutation_indices, crossover_points)
    - parent_map: list where each value is 1 (from p1), 2 (from p2), or 0 (mutation)
    - mutation_indices: list of indices where mutations occurred
    - crossover_points: list of indices where crossover occurred
    """
    # Check for None inputs
    if child is None or p1 is None or p2 is None:
        return [], [], []
    
    # Check that all inputs are dictionaries
    if not all(isinstance(x, dict) for x in [child, p1, p2]):
        return [], [], []
    
    # Access chromosomes safely with get()
    child_chromosome = child.get('chromosome', [])
    p1_chromosome = p1.get('chromosome', [])
    p2_chromosome = p2.get('chromosome', [])
    
    # Check that all chromosomes are lists/sequences
    if not all(isinstance(x, (list, tuple)) for x in [child_chromosome, p1_chromosome, p2_chromosome]):
        return [], [], []
    
    # Ensure valid chromosomes of same length
    if (not child_chromosome or not p1_chromosome or not p2_chromosome or 
        len(child_chromosome) != len(p1_chromosome) or 
        len(child_chromosome) != len(p2_chromosome)):
        return [], [], []
    
    try:
        n = len(child_chromosome)
        parent_map = [0] * n  # 0=mutation, 1=from p1, 2=from p2
        mutation_indices = []
        
        # Get the crossover point from the child data
        crossover_point = child.get('crossover_point')
        if crossover_point is None:
            # If no crossover point is recorded, estimate it
            # For permutation problems, we need to check where genes start coming from parent 2
            crossover_point = n // 2  # Default fallback
            
        # Get mutation information
        mutation_info = child.get('mutation', {})
        was_mutated = mutation_info.get('applied', False) if mutation_info else False
        mutation_positions = mutation_info.get('positions', ()) if was_mutated else ()
        
        # First handle mutation positions
        if was_mutated and mutation_positions:
            for pos in mutation_positions:
                if 0 <= pos < n:
                    parent_map[pos] = 0  # Mark as mutation
                    mutation_indices.append(pos)
        
        # For positions before crossover, should be from parent 1
        for i in range(min(crossover_point, n)):
            if i not in mutation_indices and child_chromosome[i] == p1_chromosome[i]:
                parent_map[i] = 1  # From parent 1
            elif i not in mutation_indices:
                # If not marked as mutation but doesn't match parent 1,
                # it could be a result of complex mutation/crossover interaction
                parent_map[i] = 0
                mutation_indices.append(i)
        
        # For positions after crossover, should be from parent 2 but not necessarily at same positions
        for i in range(crossover_point, n):
            if i in mutation_indices:
                continue  # Already marked as mutation
                
            # For permutation problems, check if gene exists in parent 2 (at any position)
            if child_chromosome[i] in p2_chromosome:
                parent_map[i] = 2  # From parent 2 (some position)
            else:
                # If not in parent 2, must be mutation
                parent_map[i] = 0
                mutation_indices.append(i)
        
        # Find crossover points - just one for permutation crossover
        crossover_points = [crossover_point] if 0 < crossover_point < n else []
        
        return parent_map, mutation_indices, crossover_points
    
    except (IndexError, TypeError) as e:
        # Handle any other errors that might arise during processing
        print(f"Error in get_crossover_info: {e}")
        return [], [], []

# Modified function to draw an enhanced filho board with crossover and mutation indicators
def draw_filho_board(screen, child_data, p1_data, p2_data, n, position, title, fitness):
    cell_size = calculate_cell_size(n)
    current_board_size = cell_size * n
    board_rect = pygame.Rect(position[0], position[1] + INFO_HEIGHT, current_board_size, current_board_size)

    # Draw title in bold style
    font_title = pygame.font.Font(None, 28)  # Slightly larger font
    
    # Create title
    display_title = "FILHO"
    title_surf = font_title.render(display_title, True, TITLE_COLOR)
    title_rect = title_surf.get_rect(center=(board_rect.centerx, position[1] + INFO_HEIGHT // 2))
    screen.blit(title_surf, title_rect)

    # Draw board outline with rounded corners
    pygame.draw.rect(screen, BLACK, board_rect, 1, border_radius=5)

    child_chromosome = None
    if isinstance(child_data, dict):
        child_chromosome = child_data.get('chromosome')

    if child_chromosome is None or n <= 0:
        pygame.draw.rect(screen, LIGHT_SQUARE, board_rect.inflate(-2, -2), border_radius=5)
        return board_rect
    
    # Safety checks for valid parent data
    if p1_data is None or p2_data is None or not isinstance(p1_data, dict) or not isinstance(p2_data, dict):
        # Fall back to regular drawing if parents are invalid
        for r in range(n):
            for c in range(n):
                color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, color, (board_rect.left + c * cell_size, board_rect.top + r * cell_size, cell_size, cell_size))
                
        # Draw queens without crossover/mutation coloring
        queen_radius = max(1, cell_size // 2 - 4)
        for col, row in enumerate(child_chromosome):
            if 0 <= row < n:
                center_x = board_rect.left + col * cell_size + cell_size // 2
                center_y = board_rect.top + row * cell_size + cell_size // 2
                pygame.draw.circle(screen, QUEEN_COLOR, (center_x, center_y), queen_radius)
                highlight_radius = max(1, queen_radius // 3)
                highlight_offset = max(1, queen_radius // 4)
                pygame.draw.circle(screen, (255, 100, 100), 
                                (center_x - highlight_offset, center_y - highlight_offset), 
                                highlight_radius)
        return board_rect
    
    # Get crossover information 
    parent_map, mutation_indices, crossover_points = get_crossover_info(child_data, p1_data, p2_data)
    
    # Draw the board squares
    for r in range(n):
        for c in range(n):
            color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (board_rect.left + c * cell_size, board_rect.top + r * cell_size, cell_size, cell_size))

    # Draw crossover point indicators (vertical lines between cells)
    for crossover_point in crossover_points:
        x_pos = board_rect.left + crossover_point * cell_size
        line_start = (x_pos, board_rect.top - 5)
        line_end = (x_pos, board_rect.bottom + 5)
        pygame.draw.line(screen, CROSSOVER_MARKER_COLOR, line_start, line_end, 2)
        
        # Draw small triangular markers at top and bottom
        triangle_size = 5
        pygame.draw.polygon(screen, CROSSOVER_MARKER_COLOR, [
            (x_pos, board_rect.top - triangle_size * 2),
            (x_pos - triangle_size, board_rect.top - triangle_size),
            (x_pos + triangle_size, board_rect.top - triangle_size)
        ])
        pygame.draw.polygon(screen, CROSSOVER_MARKER_COLOR, [
            (x_pos, board_rect.bottom + triangle_size * 2),
            (x_pos - triangle_size, board_rect.bottom + triangle_size),
            (x_pos + triangle_size, board_rect.bottom + triangle_size)
        ])

    # Draw queens showing parent origin with different colors
    queen_radius = max(1, cell_size // 2 - 4)
    for col, row in enumerate(child_chromosome):
        if 0 <= row < n:
            center_x = board_rect.left + col * cell_size + cell_size // 2
            center_y = board_rect.top + row * cell_size + cell_size // 2
            
            # Choose color based on parent source
            if col < len(parent_map):
                if parent_map[col] == 1:
                    queen_color = P1_COLOR
                elif parent_map[col] == 2:
                    queen_color = P2_COLOR
                else:
                    queen_color = MUTATION_COLOR
            else:
                queen_color = QUEEN_COLOR
            
            # Draw the queen
            pygame.draw.circle(screen, queen_color, (center_x, center_y), queen_radius)
            
            # Add a highlight effect
            highlight_radius = max(1, queen_radius // 3)
            highlight_offset = max(1, queen_radius // 4)
            
            # Lighter version of the queen color for highlight
            if queen_color == P1_COLOR:
                highlight_color = (150, 200, 255)  # Lighter blue
            elif queen_color == P2_COLOR:
                highlight_color = (255, 180, 180)  # Lighter red
            elif queen_color == MUTATION_COLOR:
                highlight_color = (255, 255, 150)  # Lighter yellow
            else:
                highlight_color = (255, 180, 180)  # Default
                
            pygame.draw.circle(screen, highlight_color, 
                             (center_x - highlight_offset, center_y - highlight_offset), 
                             highlight_radius)
            
            # For mutations, add a special indicator
            if col in mutation_indices:
                # Draw a star pattern or other indicator around the queen
                star_radius = queen_radius + 3
                pygame.draw.circle(screen, MUTATION_COLOR, (center_x, center_y), star_radius, 1)

    # Add a legend
    legend_font = pygame.font.Font(None, 18)
    legend_y = board_rect.bottom + 15
    
    # From Parent 1
    p1_legend_text = legend_font.render("From PAI 1", True, TITLE_COLOR)
    p1_legend_rect = p1_legend_text.get_rect(left=board_rect.left, top=legend_y)
    screen.blit(p1_legend_text, p1_legend_rect)
    pygame.draw.circle(screen, P1_COLOR, 
                     (p1_legend_rect.right + 10, p1_legend_rect.centery), 
                     5)
    
    # From Parent 2
    p2_legend_text = legend_font.render("From PAI 2", True, TITLE_COLOR)
    p2_legend_rect = p2_legend_text.get_rect(left=p1_legend_rect.right + 25, top=legend_y)
    screen.blit(p2_legend_text, p2_legend_rect)
    pygame.draw.circle(screen, P2_COLOR, 
                     (p2_legend_rect.right + 10, p2_legend_rect.centery), 
                     5)
    
    # Mutation
    mutation_legend_text = legend_font.render("Mutation", True, TITLE_COLOR)
    mutation_legend_rect = mutation_legend_text.get_rect(left=p2_legend_rect.right + 25, top=legend_y)
    screen.blit(mutation_legend_text, mutation_legend_rect)
    pygame.draw.circle(screen, MUTATION_COLOR, 
                     (mutation_legend_rect.right + 10, mutation_legend_rect.centery), 
                     5)
    
    # Crossover
    if crossover_points:
        crossover_legend_text = legend_font.render("Crossover", True, TITLE_COLOR)
        crossover_legend_rect = crossover_legend_text.get_rect(left=mutation_legend_rect.right + 25, top=legend_y)
        screen.blit(crossover_legend_text, crossover_legend_rect)
        pygame.draw.line(screen, CROSSOVER_MARKER_COLOR, 
                       (crossover_legend_rect.right + 5, crossover_legend_rect.centery - 5),
                       (crossover_legend_rect.right + 15, crossover_legend_rect.centery + 5), 
                       2)
    
    return board_rect
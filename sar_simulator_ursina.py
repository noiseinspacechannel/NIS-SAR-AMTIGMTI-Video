from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader
from sar_scene_data import SceneFetcher, MATERIALS, hex_to_rgb, SCENE_MODELS
import numpy as np
# --- Configuration ---
CENTER_LAT = 47.57881067971553
CENTER_LON = -122.4071785501292
TERRAIN_SIZE = 800
GRID_SIZE = 60

app = Ursina()
window.title = "SAR Simulator (Ursina)"
window.color = color.gray # Grey background per request
window.vsync = True
window.fullscreen = False

# --- Scene Setup ---
fetcher = SceneFetcher(CENTER_LAT, CENTER_LON, 500, GRID_SIZE, TERRAIN_SIZE)

# 1. Terrain Generation
print("Generating Terrain...")
x_coords, y_coords, elevations = fetcher.fetch_elevation_grid()

if x_coords is None:
    # Fallback procedural
    x_coords = np.linspace(-TERRAIN_SIZE, TERRAIN_SIZE, GRID_SIZE)
    y_coords = np.linspace(-TERRAIN_SIZE, TERRAIN_SIZE, GRID_SIZE)
    elevations = np.zeros((GRID_SIZE, GRID_SIZE))

fetcher.elevation_data = {'x': x_coords, 'y': y_coords, 'z': elevations}

# Create Mesh
# Ursina Mesh requires a flat list of vertices.
# We will create quads for grid cells.
vertices = []
uvs = []

# --- Dual-Mesh Generation (Land vs Water) ---
# Create distinct meshes for Land and Water based on elevation data.
# This ensures finite water that respects scene boundaries and OSM geography (via elevation).

land_verts = []
land_colors = []
water_verts = []
water_colors = []

# Iterate grid to build triangles
for y in range(GRID_SIZE - 1):
    for x in range(GRID_SIZE - 1):
        # Grid indices
        i1 = y * GRID_SIZE + x
        i2 = y * GRID_SIZE + (x + 1)
        i3 = (y + 1) * GRID_SIZE + x
        i4 = (y + 1) * GRID_SIZE + (x + 1)
        
        # Elevations
        e1, e2 = elevations.flatten()[i1], elevations.flatten()[i2]
        e3, e4 = elevations.flatten()[i3], elevations.flatten()[i4]
        
        # Coordinates
        v1 = Vec3(x_coords[x], e1, y_coords[y])
        v2 = Vec3(x_coords[x+1], e2, y_coords[y])
        v3 = Vec3(x_coords[x], e3, y_coords[y+1])
        v4 = Vec3(x_coords[x+1], e4, y_coords[y+1])
        
        # Helper to classify a quad
        avg_h = (e1+e2+e3+e4)/4.0
        is_water = avg_h < 1.0
        
        if is_water:
            # Generate WATER Mesh (Flat at z=0)
            # Use original X,Z coordinates but set Y (elevation) to 0
            w1 = Vec3(v1.x, 0, v1.z)
            w2 = Vec3(v2.x, 0, v2.z)
            w3 = Vec3(v3.x, 0, v3.z)
            w4 = Vec3(v4.x, 0, v4.z)
            
            # Quad -> 2 Triangles (CCW Winding for Upward Normals)
            # Tri 1: 1-2-3
            # Tri 2: 2-4-3
            water_verts.extend([w1, w2, w3, w2, w4, w3])
            water_colors.extend([color.hex('#1E5799')] * 6)
            
        else:
            # Generate LAND Mesh
            # Clamp underwater vertices to just below surface (-0.1) to avoid Z-fighting but close the gap
            l1 = Vec3(v1.x, max(v1.y, -0.1), v1.z)
            l2 = Vec3(v2.x, max(v2.y, -0.1), v2.z)
            l3 = Vec3(v3.x, max(v3.y, -0.1), v3.z)
            l4 = Vec3(v4.x, max(v4.y, -0.1), v4.z)
            
            # Quad -> 2 Triangles (CCW Winding)
            land_verts.extend([l1, l2, l3, l2, l4, l3])
            # Color based on height (Brown Terrain)
            c = color.hex('#8B7355')
            land_colors.extend([c]*6)

# Create Land Entity
land_mesh = Mesh(vertices=land_verts, colors=land_colors, static=True)
land_entity = Entity(model=land_mesh, collider='mesh')

# Create Water Entity
with open('debug_water.txt', 'w') as f:
    f.write(f"Water Verts: {len(water_verts)}\n")
    f.write(f"Land Verts: {len(land_verts)}\n")

print(f"Generated Water Verts: {len(water_verts)}")
water_mesh = Mesh(vertices=water_verts, colors=water_colors, static=True)
water_entity = Entity(model=water_mesh, collider='mesh', double_sided=True)
if len(water_verts) > 0:
    water_mesh.generate_normals()


# 2. Objects (OSM)
print("Fetching OSM Data...")
scene_data = fetcher.fetch_osm_data()

# Buildings
for b in scene_data['buildings']:
    # Get ground height
    z_ground = fetcher.get_elevation_at(b['x'], b['y'])
    
    mat_def = MATERIALS.get(b['material'], MATERIALS['building'])
    col = color.hex(mat_def['color'])
    
    # Ursina cube origin is center.
    # b['x'], b['y'] is center of footprint.
    # We need y (up) to be z_ground + height/2
    Entity(
        model='cube',
        color=col,
        position=(b['x'], z_ground + b['height']/2, b['y']),
        scale=(b['width'], b['height'], b['depth']),
        collider='box',
        texture='white_cube' # Adds edge shading for 3D look
    )

# Water / Polygons / Roads
# NOTE: We disable complex Polygon rendering (Water, Landuse) because fan triangulation
# of large, concave OSM shapes causes massive "vertical wall" artifacts.
# We rely on the Terrain Height coloring to show Water vs Land.

# Render Roads (Simple lines/tubes)
# Using Ursina's Pipe for smooth 3D paths
road_entities = []
for road in scene_data['roads']:
    pts = road['points']
    if len(pts) < 2: continue
    
    path_points = []
    for p in pts:
        # Get height + offset
        z = fetcher.get_elevation_at(p[0], p[1]) + 0.2
        path_points.append(Vec3(p[0], z, p[1])) # Correct (x, elev, y)
        
    if len(path_points) > 1:
        # Create a Pipe (tube) following the path
        Entity(model=Pipe(path=path_points, thicknesses=[1.5]), color=color.dark_gray)

# Water / Polygons
# Render Global Water Plane (from Data)
if 'water_plane' in scene_data:
    wp = scene_data['water_plane']
    mat_def = MATERIALS.get(wp['material'], MATERIALS['water'])
    Entity(
        model='plane',
        scale=(wp['x_size'], 1, wp['y_size']),
        color=color.hex(mat_def['color']),
        position=(0, wp['z'], 0),
        texture='white_cube'
    )

for poly in scene_data['polygons']:
    mat_type = poly['material']
    mat_def = MATERIALS.get(mat_type, MATERIALS['default'])
    
    # Check if water
    is_water = mat_type in ('water', 'ocean', 'bay', 'coastline')
    
    # Drape ALL polygons over terrain to ensure visibility
    # PyVista draped everything. Flattening water to 0 hides it under terrain.
    
    pts = poly['points']
    if len(pts) < 3: continue
    
    from sar_scene_data import triangulate_polygon
    
    verts_ref, faces = triangulate_polygon(pts)
    if verts_ref is None: continue
    
    # Convert to Ursina vertices with Correct Orientation
    mesh_verts = []
    for p in verts_ref:
        # Drape over terrain with small offset to avoid Z-fighting
        full_z = fetcher.get_elevation_at(p[0], p[1]) + 0.5
        # Fix: Vec3(x, elev, y)
        mesh_verts.append(Vec3(p[0], full_z, p[1]))
    
    final_verts = []
    # faces = list of lists
    for face in faces:
        for idx in face:
            if idx < len(mesh_verts):
                final_verts.append(mesh_verts[idx])
        
    poly_mesh = Mesh(vertices=final_verts, static=True)
    Entity(model=poly_mesh, color=color.hex(mat_def['color']))


# --- Player ---
# "Fly Mode": Gravity off, speed adjusted
# Lowered sensitivity to 25 for "wonky" camera fix
player = FirstPersonController(speed=50, mouse_sensitivity=Vec2(25, 25), gravity=0.0)
player.position = (0, 100, 0) # Start high
player.cursor.visible = False

# --- Lighting ---
# Directional light for better 3D shading
pivot = Entity()
DirectionalLight(parent=pivot, y=2, z=3, shadows=True, rotation=(45, -45, 45))
AmbientLight(color=color.rgba(100, 100, 100, 0.1))

# --- Input Handling ---
def update():
    # Fly up/down
    if held_keys['q']:
        player.y += time.dt * player.speed
    if held_keys['e']:
        player.y -= time.dt * player.speed

def input(key):
    if key == 'escape':
        application.quit()
    if key == 'f':
        window.fullscreen = not window.fullscreen
    if key == '1':
        # Reset position
        player.position = (0, 50, 0)

    if key == '1':
        # Reset position
        player.position = (0, 50, 0)

# --- Load Static Scene Models (Aircraft) from Data ---
print("Loading Scene Models...")
for model_def in SCENE_MODELS:
    path = model_def['file']
    if os.path.exists(path):
        try:
            ox, oz = model_def['position']
            s = model_def['scale']
            h_off = model_def['height_offset']
            mat_name = model_def.get('material', 'default')
            mat_data = MATERIALS.get(mat_name, MATERIALS['default'])
            
            # Place above terrain
            ground_h = fetcher.get_elevation_at(ox, oz)
            y_pos = ground_h + h_off
            
            print(f"Loading {model_def['name']} at ({ox}, {y_pos:.1f}, {oz})")
            e = Entity(
                model=path,
                scale=s,
                position=(ox, y_pos, oz),
                rotation=model_def['rotation'],
                collider='box',
                double_sided=True,
                color=color.hex(mat_data['color']), # Apply material color visual
                shader=lit_with_shadows_shader
            )
            # Tag entity for identification if needed
            e.material_id = mat_name
            
        except Exception as e:
            print(f"Failed to load {model_def['name']}: {e}")
    else:
        print(f"Model file not found: {path}")

print("Starting Ursina App...")
app.run()

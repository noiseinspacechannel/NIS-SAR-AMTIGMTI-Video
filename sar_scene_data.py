import requests
import math
import random
import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter

# --- Comprehensive Materials Dictionary ---
MATERIALS = {
    # Natural
    'water': {'color': '#4A90D9', 'roughness': 0.05, 'dielectric': 80.0},
    'ocean': {'color': '#1E5799', 'roughness': 0.02, 'dielectric': 80.0},
    'coastline': {'color': '#4A90D9', 'roughness': 0.05, 'dielectric': 80.0},
    'bay': {'color': '#4A90D9', 'roughness': 0.05, 'dielectric': 80.0},
    'sand': {'color': '#F5DEB3', 'roughness': 0.7, 'dielectric': 3.0},
    'beach': {'color': '#FFF8DC', 'roughness': 0.7, 'dielectric': 3.0},
    'wood': {'color': '#228B22', 'roughness': 0.9, 'dielectric': 5.0},
    'tree_row': {'color': '#228B22', 'roughness': 0.9, 'dielectric': 5.0},
    'scrub': {'color': '#6B8E23', 'roughness': 0.85, 'dielectric': 5.0},
    'grassland': {'color': '#90EE90', 'roughness': 0.8, 'dielectric': 5.0},
    'wetland': {'color': '#6B8E6B', 'roughness': 0.6, 'dielectric': 40.0},
    'bare_rock': {'color': '#A0A0A0', 'roughness': 0.9, 'dielectric': 7.0},
    
    # Landuse
    'residential': {'color': '#DCDCDC', 'roughness': 0.7, 'dielectric': 4.0},
    'commercial': {'color': '#F5DEB3', 'roughness': 0.6, 'dielectric': 4.0},
    'industrial': {'color': '#D8BFD8', 'roughness': 0.5, 'dielectric': 4.0},
    'retail': {'color': '#FFD0D0', 'roughness': 0.6, 'dielectric': 4.0},
    'grass': {'color': '#90EE90', 'roughness': 0.8, 'dielectric': 5.0},
    'forest': {'color': '#006400', 'roughness': 0.9, 'dielectric': 5.0},
    'farmland': {'color': '#EEE8AA', 'roughness': 0.75, 'dielectric': 4.0},
    'farmyard': {'color': '#D2B48C', 'roughness': 0.7, 'dielectric': 4.0},
    'orchard': {'color': '#9ACD32', 'roughness': 0.85, 'dielectric': 5.0},
    'meadow': {'color': '#98FB98', 'roughness': 0.8, 'dielectric': 5.0},
    'cemetery': {'color': '#AACBAF', 'roughness': 0.7, 'dielectric': 4.0},
    'construction': {'color': '#C8B464', 'roughness': 0.6, 'dielectric': 4.0},
    'railway': {'color': '#808080', 'roughness': 0.3, 'dielectric': 1000.0},
    
    # Leisure
    'park': {'color': '#C8FACC', 'roughness': 0.8, 'dielectric': 5.0},
    'garden': {'color': '#BDECB6', 'roughness': 0.8, 'dielectric': 5.0},
    'playground': {'color': '#CCFFFF', 'roughness': 0.6, 'dielectric': 4.0},
    'pitch': {'color': '#89D689', 'roughness': 0.75, 'dielectric': 5.0},
    'golf_course': {'color': '#B5E3B5', 'roughness': 0.8, 'dielectric': 5.0},
    'swimming_pool': {'color': '#66B2FF', 'roughness': 0.1, 'dielectric': 80.0},
    
    # Infrastructure
    'parking': {'color': '#F7EFCE', 'roughness': 0.3, 'dielectric': 6.0},
    'road': {'color': '#333333', 'roughness': 0.2, 'dielectric': 6.0},
    'highway': {'color': '#333333', 'roughness': 0.2, 'dielectric': 6.0},
    'footway': {'color': '#AAAAAA', 'roughness': 0.3, 'dielectric': 5.0},
    'path': {'color': '#D2B48C', 'roughness': 0.5, 'dielectric': 4.0},
    
    # Buildings
    'building': {'color': '#D9D0C9', 'roughness': 0.7, 'dielectric': 4.0},
    
    # Default/Terrain
    'default': {'color': '#C0C0C0', 'roughness': 0.5, 'dielectric': 4.0},
    'ground': {'color': '#8FBC8F', 'roughness': 0.6, 'dielectric': 4.0},
    'terrain': {'color': '#8B7355', 'roughness': 0.7, 'dielectric': 4.0},
    'metal': {'color': '#FF4444', 'roughness': 0.0, 'dielectric': 1000.0},
    
    # Aircraft Materials
    'stealth_coating': {'color': '#2A2A2A', 'roughness': 0.9, 'dielectric': 2.5}, # Low reflectivity
    'aluminum': {'color': '#C0C0C0', 'roughness': 0.1, 'dielectric': 1000.0}, # High reflectivity, Shiny
    'car_paint': {'color': '#FF0000', 'roughness': 0.2, 'dielectric': 10.0}, # Glossy Paint
    'steel_armor': {'color': '#3A4030', 'roughness': 0.7, 'dielectric': 1000.0}, # Matte Green Metal
    'ship_metal': {'color': '#708090', 'roughness': 0.6, 'dielectric': 1000.0}, # Slate Grey
}

# --- Static Models (Aircraft, etc.) ---
# Defined here to be the single source of truth for SAR simulation
SCENE_MODELS = [
    {
        'name': 'F-35B Lightning II',
        'file': 'custom_models/F-35B_default.fbx',
        'position': (0, 200), # x, z (Meters)
        'height_offset': 45.0, # Altitude above ground
        'scale': 0.02,
        'material': 'stealth_coating',
        'rotation': (0, 45, 0)
    },
    {
        'name': 'Boeing 787-8',
        'file': 'custom_models/787-8.fbx',
        'position': (60, 240), # x, z
        'height_offset': 45.0,
        'scale': 0.02,
        'material': 'aluminum',
        'rotation': (0, 45, 0)
    },
    {
        'name': 'Civilian Car',
        'file': 'custom_models/car.FBX',
        'position': (40, 210), # Adjusted to align with road
        'height_offset': 0.8, # Lift slightly (tires)
        'scale': 0.02,
        'material': 'steel_armor', # Matched to tank color
        'rotation': (-90, -35, 0) # Rotate back 85 deg (50 - 85)
    },
    {
        'name': 'Main Battle Tank',
        'file': 'custom_models/tank.FBX',
        'position': (55, 225), # Spaced out along diagonal
        'height_offset': 0.8,
        'scale': 0.02,
        'material': 'steel_armor',
        'rotation': (-90, -35, 0) # Rotate back 85 deg
    },
    {
        'name': 'Small Boat',
        'file': 'custom_models/boat.FBX',
        'position': (20, 400), # ~200m past planes
        'height_offset': 0.0, # On water
        'scale': 0.02,
        'material': 'ship_metal',
        'rotation': (-90, -35, 0)
    },
    {
        'name': 'Navy Frigate',
        'file': 'custom_models/frigate.FBX',
        'position': (-70, 600), # Moved back 150m (750-150)
        'height_offset': 0.0,
        'scale': 0.02,
        'material': 'ship_metal',
        'rotation': (-90, 145, 0) # Flip 180 deg (-35 + 180)
    }
]

# --- Helpers ---
def latlon_to_meters(lat, lon, origin_lat, origin_lon):
    x = (lon - origin_lon) * (40075000 * math.cos(math.radians(origin_lat)) / 360)
    y = (lat - origin_lat) * 111320
    return x, y

def meters_to_latlon(x, y, origin_lat, origin_lon):
    lon = origin_lon + x / (40075000 * math.cos(math.radians(origin_lat)) / 360)
    lat = origin_lat + y / 111320
    return lat, lon

def hex_to_rgb(hex_code):
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16)/255.0 for i in (0, 2, 4))

def triangulate_polygon(points_2d):
    """Triangulate a 2D polygon using fan triangulation. Returns vertices and faces indices."""
    if len(points_2d) < 3:
        return None, None
    
    pts = np.array(points_2d)
    center = pts.mean(axis=0)
    n = len(pts)
    
    # Vertices: [Center, P1, P2, ..., Pn]
    vertices = np.vstack([center, pts])
    
    # Faces: [0, 1, 2], [0, 2, 3], ...
    faces = []
    for i in range(n):
        faces.append([0, i + 1, ((i + 1) % n) + 1])
        
    # Note: 1-based indexing for the ring, 0 is center
    # Correct logic:
    # i=0 -> 0, 1, 2
    # i=n-1 -> 0, n, 1
    
    # Warning: Last point wrapping needs to handle 1-based index carefully
    # The 'pts' array indices are 0..(n-1). In 'vertices', they are 1..n.
    # ((i + 1) % n) + 1 handles wrapping correctly:
    # i=0 (Pt0) -> ((1)%n)+1 = 2 (Pt1)
    # i=n-1 (Ptn-1) -> ((n)%n)+1 = 1 (Pt0)
    
    return vertices, faces

class SceneFetcher:
    def __init__(self, center_lat, center_lon, radius, grid_size, terrain_extent):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        self.grid_size = grid_size
        self.terrain_extent = terrain_extent
        self.elevation_data = {}

    def fetch_elevation_grid(self):
        """Fetch elevation data from Open-Elevation API."""
        print("Fetching elevation data...")
        
        locations = []
        x_coords = np.linspace(-self.terrain_extent, self.terrain_extent, self.grid_size)
        y_coords = np.linspace(-self.terrain_extent, self.terrain_extent, self.grid_size)
        
        for y in y_coords:
            for x in x_coords:
                lat, lon = meters_to_latlon(x, y, self.center_lat, self.center_lon)
                locations.append({'latitude': lat, 'longitude': lon})
        
        try:
            response = requests.post(
                'https://api.open-elevation.com/api/v1/lookup',
                json={'locations': locations},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()['results']
                elevations = np.array([r['elevation'] for r in results])
                elevations = elevations.reshape((self.grid_size, self.grid_size))
                
                # Smooth the terrain to remove jagged edges
                elevations = gaussian_filter(elevations, sigma=1.0)
                
                print(f"Elevation range: {elevations.min():.1f}m to {elevations.max():.1f}m")
                return x_coords, y_coords, elevations
            else:
                print(f"Elevation API error: {response.status_code}")
                return None, None, None
        except Exception as e:
            print(f"Elevation fetch failed: {e}")
            return None, None, None

    def get_elevation_at(self, x, y):
        """Get interpolated elevation at a point."""
        if not self.elevation_data:
            return 0
        
        x_coords = self.elevation_data['x']
        y_coords = self.elevation_data['y']
        z_data = self.elevation_data['z']
        
        # Simple nearest neighbor for now to match strict data
        # Ideally bilinear interpolation
        ix = int(np.interp(x, x_coords, np.arange(len(x_coords))))
        iy = int(np.interp(y, y_coords, np.arange(len(y_coords))))
        
        # Clamp
        ix = max(0, min(ix, len(x_coords)-1))
        iy = max(0, min(iy, len(y_coords)-1))
        
        return z_data[iy, ix]

    def fetch_osm_data(self):
        """Query Overpass API and return structured scene elements."""
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:30];
        (
          way["landuse"](around:{self.radius},{self.center_lat},{self.center_lon});
          way["natural"](around:{self.radius},{self.center_lat},{self.center_lon});
          way["leisure"](around:{self.radius},{self.center_lat},{self.center_lon});
          way["amenity"="parking"](around:{self.radius},{self.center_lat},{self.center_lon});
          way["building"](around:{self.radius},{self.center_lat},{self.center_lon});
          way["highway"](around:{self.radius},{self.center_lat},{self.center_lon});
        );
        (._;>;);
        out body;
        """
        print("Requesting OSM data...")
        
        scene_objects = {
            'buildings': [], # List of dicts
            'roads': [],     # List of lists of points
            'polygons': [],   # List of dicts (vertices, type)
            # 'water_plane': Removed. Generated from elevation grid.
        }

        try:
            resp = requests.get(overpass_url, params={'data': query}, headers={'User-Agent': 'SAR_Simulator_Data/1.0'})
            data = resp.json()
            
            nodes = {n['id']: (n['lat'], n['lon']) for n in data['elements'] if n['type'] == 'node'}
            
            for el in data['elements']:
                if el['type'] == 'way' and 'nodes' in el:
                    valid_nodes = [nodes[nid] for nid in el['nodes'] if nid in nodes]
                    if len(valid_nodes) < 3:
                        # Could be a road segment with 2 nodes
                        if 'highway' in el.get('tags', {}) and len(valid_nodes) >= 2:
                            pass
                        else:
                            continue
                    
                    pts_2d = []
                    for lat, lon in valid_nodes:
                        x, y = latlon_to_meters(lat, lon, self.center_lat, self.center_lon)
                        pts_2d.append([x, y])
                    
                    # Clipping
                    pts_arr = np.array(pts_2d)
                    cx, cy = pts_arr[:, 0].mean(), pts_arr[:, 1].mean()
                    if abs(cx) > self.terrain_extent or abs(cy) > self.terrain_extent:
                        continue
                    
                    tags = el.get('tags', {})
                    mat_type = 'default'
                    is_building = 'building' in tags
                    is_road = 'highway' in tags
                    
                    # Determine Material
                    if is_building: mat_type = 'building'
                    elif is_road: mat_type = 'road'
                    elif 'landuse' in tags: mat_type = tags['landuse']
                    elif 'natural' in tags: mat_type = tags['natural']
                    elif 'leisure' in tags: mat_type = tags['leisure']
                    elif 'amenity' in tags: mat_type = tags['amenity']
                    
                    if mat_type not in MATERIALS:
                        if is_road: mat_type = 'road'
                        else: mat_type = 'default'

                    if is_building:
                        width = max(5, pts_arr[:, 0].max() - pts_arr[:, 0].min())
                        depth = max(5, pts_arr[:, 1].max() - pts_arr[:, 1].min())
                        height = random.randint(6, 14)
                        scene_objects['buildings'].append({
                            'x': cx,
                            'y': cy,
                            'width': width,
                            'depth': depth,
                            'height': height,
                            'material': mat_type,
                            'points': pts_2d # Store original points if we want exact shape later
                        })
                    elif is_road:
                        scene_objects['roads'].append({
                            'points': pts_2d,
                            'material': mat_type
                        })
                    else:
                        scene_objects['polygons'].append({
                            'points': pts_2d,
                            'material': mat_type
                        })
                        
        except Exception as e:
            print(f"OSM Error: {e}")
        
        return scene_objects

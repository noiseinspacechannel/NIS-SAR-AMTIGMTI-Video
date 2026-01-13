import numpy as np

def create_point_target(x, y, z, rcs, name=""):
    return {'position': [x, y, z], 'rcs': rcs, 'name': name}

def generate_car(center_pos=(0,0,0), name_prefix="Car"):
    # Dimensions: 4.5m x 1.8m x 1.4m
    # Total RCS goal: ~10 m^2
    targets = []
    cx, cy, cz = center_pos
    
    # Chassis corners
    l, w = 4.5, 1.8
    z_chassis = 0.5
    corners = [
        (l/2, w/2, z_chassis), (l/2, -w/2, z_chassis),
        (-l/2, w/2, z_chassis), (-l/2, -w/2, z_chassis)
    ]
    
    # Roof corners
    l_roof, w_roof = 2.0, 1.4
    z_roof = 1.4
    roof_corners = [
        (l_roof/2, w_roof/2, z_roof), (l_roof/2, -w_roof/2, z_roof),
        (-l_roof/2, w_roof/2, z_roof), (-l_roof/2, -w_roof/2, z_roof)
    ]
    
    # Wheels/Bumpers
    extras = [
        (l/2, 0, 0.4), (-l/2, 0, 0.4) 
    ]
    
    mid_pts = [
        (0, w/2, 0.9), (0, -w/2, 0.9) 
    ]

    for i, (lx, ly, lz) in enumerate(corners + roof_corners + extras + mid_pts):
        r = 1.0 # ~12 points * 1.0 = ~12 m^2 total (constructive interference varies)
        targets.append(create_point_target(cx+lx, cy+ly, cz+lz, r, f"{name_prefix}_pt{i}"))
        
    return targets

def generate_tank(center_pos=(0,0,0), name_prefix="Tank"):
    targets = []
    cx, cy, cz = center_pos
    
    l, w, h = 8.0, 3.6, 1.5
    hull_pts = [
        (l/2, w/2, h), (l/2, -w/2, h), (-l/2, w/2, h), (-l/2, -w/2, h),
        (l/2, w/2, 0.5), (l/2, -w/2, 0.5), (-l/2, w/2, 0.5), (-l/2, -w/2, 0.5)
    ]
    
    t_rad = 1.5
    z_turret = 2.3
    turret_pts = [
        (0, 0, z_turret),
        (t_rad, 0, z_turret-0.3), (-t_rad, 0, z_turret-0.3),
        (0, t_rad, z_turret-0.3), (0, -t_rad, z_turret-0.3)
    ]
    
    gun_pts = [
        (l/2 + 1.0, 0, z_turret-0.5), (l/2 + 3.0, 0, z_turret-0.5), (l/2 + 5.0, 0, z_turret-0.5)
    ]
    
    mid_hull_pts = [
        (0, w/2, 1.0), (0, -w/2, 1.0)
    ]
    
    for i, (lx, ly, lz) in enumerate(hull_pts + turret_pts + gun_pts + mid_hull_pts):
        r = 5.0 
        targets.append(create_point_target(cx+lx, cy+ly, cz+lz, r, f"{name_prefix}_pt{i}"))
        
    return targets

def generate_fighter_jet(center_pos=(0,0,0), name_prefix="Jet4Gen", rcs_scale=1.0):
    targets = []
    cx, cy, cz = center_pos
    
    body_pts = [
        (7.5, 0, 0), (5.0, 0, 1.0), (-6.0, 0, 1.0), 
        (-7.0, 0, 0.5), (-6.0, 0, 2.5),
    ]
    
    wing_pts = [
        (0, 2.0, 0), (0, -2.0, 0), (-3.0, 5.0, 0), (-3.0, -5.0, 0),
        (-4.0, 2.5, 0), (-4.0, -2.5, 0)
    ]
    
    stab_pts = [
        (-6.5, 2.0, 0), (-6.5, -2.0, 0)
    ]
    
    for i, (lx, ly, lz) in enumerate(body_pts + wing_pts + stab_pts):
        r = 10.0 * rcs_scale 
        targets.append(create_point_target(cx+lx, cy+ly, cz+lz, r, f"{name_prefix}_pt{i}"))

    return targets

def generate_f35(center_pos=(0,0,0), name_prefix="F35"):
    return generate_fighter_jet(center_pos, name_prefix, rcs_scale=0.01)

def generate_destroyer(center_pos=(0,0,0), name_prefix="Destroyer"):
    # Arleigh Burke Flight I approx: 154m x 20m
    # Total RCS goal: ~50,000 m^2 (Typical large ship)
    targets = []
    cx, cy, cz = center_pos
    
    length = 154.0
    width = 20.0
    
    # Grid of points along the hull - REVERTED TO LOW RES
    rows = 5
    cols = 3 
    
    x_steps = np.linspace(-length/2, length/2, rows)
    y_steps = np.linspace(-width/2, width/2, cols)
    
    # Hull points (15 * 2 = 30 points)
    for x in x_steps:
        for y in y_steps:
            # Hull and Deck - massive vertical surfaces
            targets.append(create_point_target(cx+x, cy+y, cz+1, 1000.0, f"{name_prefix}_hull"))
            targets.append(create_point_target(cx+x, cy+y, cz+6, 1000.0, f"{name_prefix}_deck"))
            
    # Superstructure (strong corner reflectors)
    bridge_x = length * 0.2
    targets.append(create_point_target(cx+bridge_x, cy, cz+15, 5000.0, f"{name_prefix}_bridge"))
    
    mast_x = length * 0.1
    targets.append(create_point_target(cx+mast_x, cy, cz+25, 3000.0, f"{name_prefix}_mast"))
    
    stack_x = -length * 0.1
    targets.append(create_point_target(cx+stack_x, cy, cz+12, 3000.0, f"{name_prefix}_stack"))
    
    bow_x = length/2.0 + 10.0
    targets.append(create_point_target(cx+bow_x, cy, cz+6, 1000.0, f"{name_prefix}_bow"))
    
    stern_x = -length/2.0 - 5.0
    targets.append(create_point_target(cx+stern_x, cy, cz+6, 1000.0, f"{name_prefix}_stern"))
    
    return targets

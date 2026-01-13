from ursina import *

# === Configuration ===
app = Ursina()
window.title = "XP SAR Scan"
window.fullscreen = True
window.color = color.black
window.fps_counter.enabled = False # Remove FPS
window.entity_counter.enabled = False # Remove Entity Count
window.collider_counter.enabled = False # Remove Collider Count
window.exit_button.visible = False # Remove default exit button
# === Image & Shader ===
# Load texture
xp_tex = load_texture('bliss.jpg')

# Pixelation Shader
sar_scan_shader = Shader(language=Shader.GLSL, vertex='''
#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uv = p3d_MultiTexCoord0;
}
''', fragment='''
#version 140
uniform sampler2D p3d_Texture0;
in vec2 uv;
out vec4 fragColor;

uniform float scan_rows_total; 
uniform float current_row_idx; 
uniform float scan_progress_x; 
uniform float aspect_ratio; 

void main() {
    // Grid Setup
    float rows = scan_rows_total;
    float cols = scan_rows_total * aspect_ratio; 
    
    // Explicitly use aspect_ratio to prevent optimization culling
    if (aspect_ratio < 0.001) cols = 20.0; 
    
    // Fixed columns for consistent pixelation
    cols = 20.0;
    
    float row_id = floor((1.0 - uv.y) * rows); 
    float col_id = floor(uv.x * cols);
    
    vec4 tex_color = texture(p3d_Texture0, uv);
    
    // Grayscale
    float gray = dot(tex_color.rgb, vec3(0.299, 0.587, 0.114));
    vec4 final_col = vec4(gray, gray, gray, 1.0);
    
    // === Pixelation Logic ===
    bool is_processed = false;
    
    if (row_id < current_row_idx) {
        is_processed = true;
    }
    else if (row_id == current_row_idx) {
        float current_scan_col = floor(scan_progress_x * cols);
        if (col_id < current_scan_col) {
            is_processed = true;
        }
    }
    
    if (is_processed) {
        vec2 block_size = vec2(1.0/cols, 1.0/rows);
        vec2 center_uv = vec2((col_id + 0.5) * block_size.x, 1.0 - (row_id + 0.5) * block_size.y);
        
        vec4 block_sample = texture(p3d_Texture0, center_uv);
        float block_gray = dot(block_sample.rgb, vec3(0.299, 0.587, 0.114));
        final_col = vec4(block_gray, block_gray, block_gray, 1.0);
    }
    
    // === Beam Overlay ===
    float beam_intensity = 0.0;
    
    if (abs(row_id - current_row_idx) < 0.5) {
        float row_h_uv = 1.0 / rows;
        // Square Beam Calc
        // Screen W * beam_w_uv = Screen H * row_h_uv
        // W/H = AspectRatio
        // beam_w_uv * Aspect = row_h_uv
        // beam_w_uv = row_h_uv / Aspect
        float beam_width_uv = row_h_uv * (1.0 / aspect_ratio) * 0.5; 
        
        if (abs(uv.x - scan_progress_x) < beam_width_uv) {
             beam_intensity = 1.0; 
        }
    }
    
    if (beam_intensity > 0.0) {
        final_col.rgb += vec3(0.0, 1.0, 0.0) * beam_intensity; 
    }
    
    fragColor = final_col;
}
''')

# === Scene Setup ===
# Aspect Ratio of 800x600 (XP) is 1.33? 
# Bliss is usually 4:3 -> 1.33
aspect = 1.33 
quad_height = 10
quad_width = quad_height * aspect

display_quad = Entity(model='quad', texture=xp_tex, scale=(quad_width, quad_height), shader=sar_scan_shader)

# Initialize Uniforms IMMEDIATELY to prevent Shader Error
display_quad.set_shader_input('scan_rows_total', 20.0)
display_quad.set_shader_input('current_row_idx', -1.0)
display_quad.set_shader_input('scan_progress_x', 0.0)
display_quad.set_shader_input('aspect_ratio', aspect)
display_quad.set_shader_input('time_sec', 0.0)

# === Logic ===
total_time = 40.0
rows = 20
time_per_row = total_time / rows

run_time = 0.0

def update():
    global run_time
    run_time += time.dt
    
    if run_time < total_time:
        row_idx = int(run_time / time_per_row)
        row_time = run_time % time_per_row
        progress = row_time / time_per_row
        
        display_quad.set_shader_input('current_row_idx', float(row_idx))
        display_quad.set_shader_input('scan_progress_x', progress)
        display_quad.set_shader_input('time_sec', run_time)
        display_quad.set_shader_input('aspect_ratio', aspect) # Keep setting it just in case
    else:
        display_quad.set_shader_input('current_row_idx', 999.0)

def input(key):
    if key == 'escape':
        application.quit()

app.run()

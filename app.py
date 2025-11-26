import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Tensile Test Sim Ultimate")

st.title("3D Tensile Test: Fracture & Performance Mode")


# --- 1. Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Material Properties")
    
    E_mod = st.slider("Young's Modulus (E) [MPa]", 500, 5000, 2000, step=100)
    Sig_y = st.slider("Yield Strength [MPa]", 20, 100, 40)
    Sig_uts = st.slider("Ult. Tensile Strength (UTS)", 50, 150, 70)
    
    st.divider()
    
    st.header("🎬 Animation Settings")
    n_frames = st.slider("Smoothness (Frames)", 20, 80, 50) 
    speed = st.slider("Animation Time (ms)", 1000, 5000, 2500)
    neck_severity = st.slider("Necking Depth", 0.3, 0.9, 0.7)
    
    resolution = 35 

# --- 2. Physics Engine ---

def calculate_physics(E, Sy, Suts):
    eps = np.linspace(0, 0.55, n_frames)
    stress = np.zeros_like(eps)
    
    e_yield = Sy / E
    e_uts = 0.25 
    
    mask_elas = eps <= e_yield
    stress[mask_elas] = eps[mask_elas] * E
    
    mask_plas = (eps > e_yield) & (eps <= e_uts)
    if np.any(mask_plas):
        e_norm = (eps[mask_plas] - e_yield) / (e_uts - e_yield)
        stress[mask_plas] = Sy + (Suts - Sy) * (2*e_norm - e_norm**2)
        
    mask_neck = eps > e_uts
    if np.any(mask_neck):
        e_n = eps[mask_neck]
        stress[mask_neck] = Suts * (1 - 0.5 * ((e_n - e_uts)/(0.55 - e_uts))**2)
        
    return eps, stress, e_uts

def get_dogbone_mesh(strain, e_uts, is_broken=False):
    """
    Returns mesh data. Only generates Top Fracture Cap.
    """
    # 1. Base Grid
    z_lin = np.linspace(-1.2, 1.2, resolution)
    theta_lin = np.linspace(0, 2*np.pi, resolution)
    Z_grid, Theta_grid = np.meshgrid(z_lin, theta_lin)
    
    # 2. Geometry Profile
    shoulder_mask = 0.5 * (1 + np.tanh(12 * (np.abs(Z_grid) - 0.6)))
    r_gauge = 0.12
    r_grip = 0.28
    R_base = r_gauge + (r_grip - r_gauge) * shoulder_mask
    
    # 3. Deform
    stretch_factor = 1.0 - shoulder_mask 
    Z_deformed = Z_grid * (1 + strain * stretch_factor)
    
    nu = 0.33
    R_deformed = R_base * (1 - nu * strain * stretch_factor)
    
    # 4. Necking & Color
    color_intensity = np.zeros_like(Z_deformed)
    
    if strain > e_uts:
        neck_progress = (strain - e_uts) / (0.55 - e_uts)
        neck_progress = min(neck_progress, 1.0)
        
        notch = np.exp(- (Z_deformed / 0.2)**2)
        reduction = (neck_severity * neck_progress) * notch
        R_deformed = R_deformed * (1 - reduction)
        color_intensity = notch * neck_progress

    # --- 5. FRACTURE LOGIC ---
    
    # Init empty arrays for Top Cap only
    Xc_frac_top = np.full((2, resolution), np.nan)
    Yc_frac_top = np.full((2, resolution), np.nan)
    Zc_frac_top = np.full((2, resolution), np.nan)
    
    # Define Cut Threshold
    cut_half_width = 0.05
    
    if is_broken:
        gap_size = 0.15
        
        # 1. Apply Shift to Main Mesh
        top_part_mask = Z_grid > 0
        bot_part_mask = Z_grid < 0
        Z_deformed[top_part_mask] += gap_size
        Z_deformed[bot_part_mask] -= gap_size
        
        # 2. Hide Center Region
        hide_mask = (Z_grid > -cut_half_width) & (Z_grid < cut_half_width)
        R_deformed[hide_mask] = np.nan 
        
        # 3. Find Top Lip Only
        valid_top_indices = np.where(z_lin >= cut_half_width)[0]
        
        if len(valid_top_indices) > 0:
            idx_top = valid_top_indices[0]  # The first valid row of top piece
            
            # Extract EXACT coordinates
            z_lip_top = Z_deformed[0, idx_top]
            r_lip_top = R_deformed[0, idx_top]
            
            # Generate Top Cap
            r_vals_top = np.linspace(0, r_lip_top, 2)
            R_cap_t, T_cap_t = np.meshgrid(r_vals_top, theta_lin)

            Xc_frac_top = R_cap_t * np.cos(T_cap_t)
            Yc_frac_top = R_cap_t * np.sin(T_cap_t)
            Zc_frac_top = np.full_like(Xc_frac_top, z_lip_top)

    # Convert Cylindrical to Cartesian
    X = R_deformed * np.cos(Theta_grid)
    Y = R_deformed * np.sin(Theta_grid)
    Z = Z_deformed
    
    # --- Grip Caps ---
    z_grip_top = Z_deformed[0, -1]
    z_grip_bot = Z_deformed[0, 0]
    r_grip = R_deformed[0, -1]
    
    r_cap = np.linspace(0, r_grip, 2)
    R_cap, T_cap = np.meshgrid(r_cap, theta_lin)
    
    Xc_cap = R_cap * np.cos(T_cap)
    Yc_cap = R_cap * np.sin(T_cap)
    Zc_top = np.full_like(Xc_cap, z_grip_top)
    Zc_bot = np.full_like(Xc_cap, z_grip_bot)

    # Return only top fracture cap data
    return X, Y, Z, Xc_cap, Yc_cap, Zc_top, Zc_bot, Xc_frac_top, Yc_frac_top, Zc_frac_top, color_intensity

# --- 3. Pre-computation ---

eps_data, stress_data, e_uts_val = calculate_physics(E_mod, Sig_y, Sig_uts)

frames = []
break_index = int(len(eps_data) * 0.88) 

colorscale_custom = [
    [0.0, 'rgb(200, 200, 205)'], 
    [0.2, 'rgb(200, 200, 205)'],
    [0.5, 'rgb(255, 140, 0)'],   
    [1.0, 'rgb(220, 0, 0)']      
]

for k in range(len(eps_data)):
    e_k = eps_data[k]
    s_k = stress_data[k]
    
    broken_state = (k >= break_index)
    if broken_state: s_k = 0
        
    (X, Y, Z, Xc, Yc, Zt, Zb, Xft, Yft, Zft, Colors) = get_dogbone_mesh(e_k, e_uts_val, is_broken=broken_state)
    
    x_trace = list(eps_data[:k+1])
    y_trace = list(stress_data[:k+1])
    if broken_state: y_trace[-1] = 0
    
    frames.append(go.Frame(
        data=[
            go.Surface(x=X, y=Y, z=Z, surfacecolor=Colors), 
            go.Surface(x=Xc, y=Yc, z=Zt), 
            go.Surface(x=Xc, y=Yc, z=Zb), 
            go.Surface(x=Xft, y=Yft, z=Zft), # Only Top Fracture Surface
            go.Scatter(x=x_trace, y=y_trace),
            go.Scatter(x=[e_k], y=[s_k])
        ],
        name=str(k)
    ))

# --- 4. Plot Initialization ---

(X0, Y0, Z0, Xc0, Yc0, Zt0, Zb0, Xft0, Yft0, Zft0, C0) = get_dogbone_mesh(0, e_uts_val)

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "surface"}, {"type": "xy"}]],
    column_widths=[0.4, 0.6],
    subplot_titles=("3D Specimen", "Stress-Strain Curve")
)

lighting_fx = dict(ambient=0.4, diffuse=0.9, roughness=0.1, specular=0.3)

# Body
fig.add_trace(go.Surface(x=X0, y=Y0, z=Z0, surfacecolor=C0, 
                         colorscale=colorscale_custom, cmin=0, cmax=1, showscale=False,
                         lighting=lighting_fx, name="Body"), 1, 1)

# Grips
fig.add_trace(go.Surface(x=Xc0, y=Yc0, z=Zt0, colorscale=[[0, 'rgb(180,180,180)'], [1, 'rgb(180,180,180)']], 
                         showscale=False, opacity=1, lighting=lighting_fx, name="Grip Cap"), 1, 1)
fig.add_trace(go.Surface(x=Xc0, y=Yc0, z=Zb0, colorscale=[[0, 'rgb(180,180,180)'], [1, 'rgb(180,180,180)']], 
                         showscale=False, opacity=1, lighting=lighting_fx, name="Grip Cap"), 1, 1)

# Fracture Cap (Top Only)
fig.add_trace(go.Surface(x=Xft0, y=Yft0, z=Zft0, colorscale=[[0, 'rgb(60,60,60)'], [1, 'rgb(60,60,60)']], 
                         showscale=False, opacity=1, name="Fracture Surface"), 1, 1)

# Graph
fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='royalblue', width=4), name='Path'), 1, 2)
fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='red', size=15, line=dict(color='white', width=2)), name='Tip'), 1, 2)

fig.add_annotation(x=eps_data[break_index], y=stress_data[break_index], text="Fracture X", 
                   showarrow=True, arrowhead=1, row=1, col=2, font=dict(color="red"))

fig.update(frames=frames)

fig.update_layout(
    height=600,
    margin=dict(l=10, r=10, t=40, b=10),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False, range=[-2.1, 2.1]),
        aspectmode="data",
        camera=dict(eye=dict(x=2.2, y=0.4, z=0.4))
    ),
    xaxis2=dict(title="Strain (ε)", range=[-0.05, 0.6]),
    yaxis2=dict(title="Stress (MPa)", range=[0, Sig_uts*1.5]),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.05, y=1.1,
        buttons=[dict(
            label="▶ RUN TEST",
            method="animate",
            args=[None, dict(frame=dict(duration=speed/n_frames, redraw=True), 
                             fromcurrent=True, mode="immediate", transition=dict(duration=0))]
        )]
    )]
)

st.plotly_chart(fig, use_container_width=True)
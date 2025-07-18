import streamlit as st
import pandas as pd
import numpy as np
from math import sin, cos, tan, radians, degrees, sqrt, atan2, acos
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Fungsi untuk menyimpan ke Google Sheets
def save_to_gsheets(data):
    try:
        # Scope dan auth
        scope = ["https://spreadsheets.google.com/feeds", 
                "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("gsheets-key.json", scope)
        client = gspread.authorize(creds)
        
        # Buka spreadsheet (ganti dengan URL/nama spreadsheet Anda)
        sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1H1ezTfhejzScdRLCHPuM3nOBDfkUT6_pNEAvfFSchRc/edit?usp=sharing").sheet1
        
        # Format data
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data["well_name"],
            data["field_name"],
            data["company"],
            data["surface_northing"],
            data["surface_easting"],
            data["target_northing"],
            data["target_easting"],
            data["target_depth"],
            data["kop"],
            data["bur"],
            data["target_inc"],
            data["distance_to_target"],
            str(data["summary_df"].to_dict("records")),  # Simpan sebagai string JSON
            str(data["detailed_df"].to_dict("records"))
        ]
        
        # Tambahkan baris baru
        sheet.append_row(row)
        return True
    except Exception as e:
        st.error(f"Gagal menyimpan: {str(e)}")
        return False


# Helper functions
def calculate_dogleg_angle(inc1, inc2, azi1, azi2):
    """Calculate dogleg angle between two survey stations"""
    inc1_rad = radians(inc1)
    inc2_rad = radians(inc2)
    azi_diff_rad = radians(azi2 - azi1)
    
    dl_rad = acos(
        cos(inc1_rad) * cos(inc2_rad) + 
        sin(inc1_rad) * sin(inc2_rad) * cos(azi_diff_rad)
    )
    return degrees(dl_rad)

def minimum_curvature(inc1, inc2, azi1, azi2, md):
    """Minimum curvature calculations"""
    dl = radians(calculate_dogleg_angle(inc1, inc2, azi1, azi2))
    if dl < 0.0001:  # Straight section
        rf = 1.0
    else:
        rf = 2 / dl * tan(dl / 2)
    
    inc_avg = radians((inc1 + inc2) / 2)
    azi_avg = radians((azi1 + azi2) / 2)
    
    d_tvd = md * cos(inc_avg) * rf
    d_northing = md * sin(inc_avg) * cos(azi_avg) * rf
    d_easting = md * sin(inc_avg) * sin(azi_avg) * rf
    
    return d_tvd, d_northing, d_easting, rf

def calculate_trajectory(surface_northing, surface_easting, rkb_elevation, 
                         target_northing, target_easting, target_depth, 
                         kop, bur, target_inc=None):
    """Main trajectory calculation function"""
    # Calculate horizontal displacement and azimuth
    delta_n = target_northing - surface_northing
    delta_e = target_easting - surface_easting
    delta_h = sqrt(delta_n**2 + delta_e**2)
    azimuth = degrees(atan2(delta_e, delta_n)) % 360
    
    # Calculate radius of curvature
    bur_rad = bur * (np.pi/180) / 30  # Convert to rad/m
    radius = 1 / bur_rad
    
    # Initialize trajectory points
    points = []
    
    # 1. RKB Point
    points.append({
        'Parameter': 'RKB',
        'MD': 0.0,
        'TVD': 0.0,
        'Inc': 0.0,
        'Azimuth': azimuth,
        'N+': 0.0,
        'E+': 0.0,
        'Northing': surface_northing,
        'Easting': surface_easting,
        'Displacement': 0.0,
        'TVDSS': rkb_elevation,
        'BUR': 0.0
    })
    
    # 2. KOP Point (vertical section)
    points.append({
        'Parameter': 'KOP',
        'MD': kop,
        'TVD': kop,
        'Inc': 0.0,
        'Azimuth': azimuth,
        'N+': 0.0,
        'E+': 0.0,
        'Northing': surface_northing,
        'Easting': surface_easting,
        'Displacement': 0.0,
        'TVDSS': rkb_elevation - kop,
        'BUR': 0.0
    })
    
    # Use provided inclination
    inc = target_inc
    
    # 3. Build section (KOP to EOB)
    build_length = radius * radians(inc)
    eob_md = kop + build_length
    
    # Calculate displacement at EOB
    hd_eob = radius * (1 - cos(radians(inc)))
    tvd_eob = kop + radius * sin(radians(inc))
    
    # Calculate northing and easting at EOB (relative to surface)
    n_eob = hd_eob * cos(radians(azimuth))
    e_eob = hd_eob * sin(radians(azimuth))
    
    points.append({
        'Parameter': 'EOB',
        'MD': eob_md,
        'TVD': tvd_eob,
        'Inc': inc,
        'Azimuth': azimuth,
        'N+': n_eob,
        'E+': e_eob,
        'Northing': surface_northing + n_eob,
        'Easting': surface_easting + e_eob,
        'Displacement': hd_eob,
        'TVDSS': rkb_elevation - tvd_eob,
        'BUR': bur
    })
    
    # 4. Tangent section (EOB to Target)
    delta_tvd_target = (rkb_elevation - target_depth) - tvd_eob
    tangent_length = delta_tvd_target / cos(radians(inc))
    
    hd_target = hd_eob + tangent_length * sin(radians(inc))
    target_md = eob_md + tangent_length
    
    # Calculate northing and easting at Target (relative to surface)
    n_target = hd_target * cos(radians(azimuth))
    e_target = hd_target * sin(radians(azimuth))
    
    points.append({
        'Parameter': 'Target',
        'MD': target_md,
        'TVD': rkb_elevation - target_depth,
        'Inc': inc,
        'Azimuth': azimuth,
        'N+': n_target,
        'E+': e_target,
        'Northing': surface_northing + n_target,
        'Easting': surface_easting + e_target,
        'Displacement': hd_target,
        'TVDSS': target_depth,
        'BUR': 0.0
    })
    
    # 5. TD (round up to nearest 30m)
    td_md = ((int(target_md) // 30) + 1) * 30
    if td_md <= target_md:
        td_md += 30
    
    delta_md_td = td_md - target_md
    delta_tvd_td = delta_md_td * cos(radians(inc))
    delta_hd_td = delta_md_td * sin(radians(inc))
    
    tvd_td = (rkb_elevation - target_depth) + delta_tvd_td
    hd_td = hd_target + delta_hd_td
    
    n_td = hd_td * cos(radians(azimuth))
    e_td = hd_td * sin(radians(azimuth))
    
    points.append({
        'Parameter': 'TD',
        'MD': td_md,
        'TVD': tvd_td,
        'Inc': inc,
        'Azimuth': azimuth,
        'N+': n_td,
        'E+': e_td,
        'Northing': surface_northing + n_td,
        'Easting': surface_easting + e_td,
        'Displacement': hd_td,
        'TVDSS': rkb_elevation - tvd_td,
        'BUR': 0.0
    })
    
    # Generate detailed survey at 30m intervals including KOP, EOB, and Target
    detailed_survey = []
    
    # Create all possible MD points (30m intervals + key points)
    regular_intervals = list(np.arange(0, td_md + 30, 30))  # 30m intervals
    key_points = [0, kop, eob_md, target_md, td_md]  # Key points
    all_md_points = sorted(list(set(regular_intervals + key_points)))  # Combine and remove duplicates
    
    for md in all_md_points:
        if md > td_md:
            continue
            
        if md <= kop:  # Vertical section
            tvd = md
            inc = 0.0
            hd = 0.0
            n_rel = 0.0
            e_rel = 0.0
            bur_val = 0.0
            parameter = "RKB" if md == 0 else "KOP" if md == kop else "Vertical"
        elif md <= eob_md:  # Build section
            alpha = (md - kop) / radius
            inc = degrees(alpha)
            tvd = kop + radius * sin(alpha)
            hd = radius * (1 - cos(alpha))
            n_rel = hd * cos(radians(azimuth))
            e_rel = hd * sin(radians(azimuth))
            bur_val = bur
            parameter = "EOB" if md == eob_md else "Build"
        else:  # Tangent section
            inc = target_inc
            delta_md = md - eob_md
            tvd = tvd_eob + delta_md * cos(radians(inc))
            hd = hd_eob + delta_md * sin(radians(inc))
            n_rel = hd * cos(radians(azimuth))
            e_rel = hd * sin(radians(azimuth))
            bur_val = 0.0
            parameter = "Target" if md == target_md else ("TD" if md == td_md else "Tangent")
        
        detailed_survey.append({
            'MD': md,
            'TVD': tvd,
            'Inc': inc,
            'Azimuth': azimuth,
            'N+': n_rel,
            'E+': e_rel,
            'Northing': surface_northing + n_rel,
            'Easting': surface_easting + e_rel,
            'Displacement': hd,
            'TVDSS': rkb_elevation - tvd,
            'BUR': bur_val,
            'Parameter': parameter
        })
    
    return pd.DataFrame(points), pd.DataFrame(detailed_survey), hd_target, delta_h

# Streamlit UI Configuration
st.set_page_config(layout="wide")
st.title("Well Trajectory Planner")
st.markdown("**by Rigsis Drilling Team. For J-Type Well with constant Azimuth. Minimum Curvature method.**")

# Custom CSS for styling
st.markdown("""
<style>
div[data-testid="stDataFrame"] {
    width: 100% !important;
}
.dataframe th, .dataframe td {
    white-space: nowrap;
    text-align: right;
}
.distance-good {
    color: #2ecc71;
    font-weight: bold;
}
.distance-bad {
    color: #e74c3c;
    font-weight: bold;
}
.copy-button {
    float: right;
    margin-top: -40px;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

# Session state to preserve calculation results
if 'results_calculated' not in st.session_state:
    st.session_state.results_calculated = False
    st.session_state.summary_df = None
    st.session_state.detailed_df = None

# General Information
st.header("General Information")
col1, col2, col3 = st.columns(3)
with col1:
    by_well = st.text_input("By Well")
with col2:
    by_field = st.text_input("By Field")
with col3:
    by_company = st.text_input("By Company")

# Surface Point
st.header("Surface Point")
col1, col2, col3, col4 = st.columns(4)
with col1:
    surface_northing = st.number_input("Northing (m)", value=9202757.149, format="%.2f")
with col2:
    surface_easting = st.number_input("Easting (m)", value=377233.268, format="%.2f")
with col3:
    ground_level = st.number_input("Ground Level (mASL)", value=1000.00, format="%.2f")
with col4:
    rig_elevation = st.number_input("Rig Elevation (m)", value=0.00, format="%.2f")

rkb_elevation = ground_level + rig_elevation
st.markdown(f"**RKB Elevation, mASL:** {rkb_elevation:,.2f}")

# Subsurface Target
st.header("Subsurface Target")
col1, col2, col3 = st.columns(3)
with col1:
    target_northing = st.number_input("Target Northing (m)", value=9202081.409, format="%.2f")
with col2:
    target_easting = st.number_input("Target Easting (m)", value=377153.018, format="%.2f")
with col3:
    target_depth = st.number_input("Target Depth (mASL)", value=-1690.74, format="%.2f")

# Design Parameters
st.header("Design Parameters")

calculation_method = st.radio("Calculation Method", 
                            ["KOP + Inclination", "KOP + BUR", "Inclination + BUR"])

if calculation_method == "KOP + Inclination":
    col1, col2 = st.columns(2)
    with col1:
        kop = st.number_input("KOP (m)", value=500.00, format="%.2f")
    with col2:
        target_inc = st.number_input("Target Inclination (deg)", value=30.00, format="%.2f")
    bur = None
elif calculation_method == "KOP + BUR":
    col1, col2 = st.columns(2)
    with col1:
        kop = st.number_input("KOP (m)", value=500.00, format="%.2f")
    with col2:
        bur = st.number_input("BUR (deg/30m)", value=2.00, format="%.2f")
    target_inc = None
else:  # Inclination + BUR
    col1, col2 = st.columns(2)
    with col1:
        target_inc = st.number_input("Target Inclination (deg)", value=30.00, format="%.2f")
    with col2:
        bur = st.number_input("BUR (deg/30m)", value=2.00, format="%.2f")
    kop = None

if st.button("Calculate Trajectory"):
    delta_tvd = (rkb_elevation - target_depth)
    delta_h = sqrt((target_northing - surface_northing)**2 + 
                 (target_easting - surface_easting)**2)
    
    if calculation_method == "KOP + Inclination":
        # Exact solution for BUR
        alpha_rad = radians(target_inc)
        numerator = delta_h - (delta_tvd - kop) * tan(alpha_rad)
        denominator = (1 - cos(alpha_rad)) - sin(alpha_rad) * tan(alpha_rad)
        radius = numerator / denominator
        bur = degrees(1/radius) * 30  # Convert to °/30m
        
        summary_df, detailed_df, hd_target, delta_h = calculate_trajectory(
            surface_northing, surface_easting, rkb_elevation,
            target_northing, target_easting, target_depth,
            kop, bur, target_inc
        )

    elif calculation_method == "KOP + BUR":
        # Solve α numerically
        radius = 1 / (bur * (np.pi/180) / 30)
        
        def f(alpha):
            alpha_rad = radians(alpha)
            hd = radius * (1 - cos(alpha_rad))
            tvd = kop + radius * sin(alpha_rad)
            remaining_tvd = delta_tvd - tvd
            hd += remaining_tvd * tan(alpha_rad)
            return hd - delta_h
        
        # Bisection method
        alpha_low, alpha_high = 0.1, 89.9
        for _ in range(20):
            alpha_mid = (alpha_low + alpha_high)/2
            if f(alpha_mid) * f(alpha_low) < 0:
                alpha_high = alpha_mid
            else:
                alpha_low = alpha_mid
        inc = (alpha_low + alpha_high)/2
        
        summary_df, detailed_df, hd_target, delta_h = calculate_trajectory(
            surface_northing, surface_easting, rkb_elevation,
            target_northing, target_easting, target_depth,
            kop, bur, inc
        )

    else:  # Inclination + BUR
        radius = 1 / (bur * (np.pi/180) / 30)
        alpha_rad = radians(target_inc)
        
        kop = delta_tvd - delta_h * (1/tan(alpha_rad)) - radius * ((1 - cos(alpha_rad))/sin(alpha_rad))
        
        summary_df, detailed_df, hd_target, delta_h = calculate_trajectory(
            surface_northing, surface_easting, rkb_elevation,
            target_northing, target_easting, target_depth,
            kop, bur, target_inc
        )
    
    # Store results in session state
    st.session_state.results_calculated = True
    st.session_state.summary_df = summary_df
    st.session_state.detailed_df = detailed_df
    st.session_state.distance_to_target = abs(delta_h - hd_target)
    # Di dalam blok if st.button("Calculate Trajectory"): setelah store results
    st.session_state.surface_northing = surface_northing
    st.session_state.surface_easting = surface_easting
    st.session_state.target_northing = target_northing
    st.session_state.target_easting = target_easting
    st.session_state.target_depth = target_depth
    st.session_state.rkb_elevation = rkb_elevation

    st.session_state.update({
        'results_calculated': True,
        'summary_df': summary_df,
        'detailed_df': detailed_df,
        'distance_to_target': abs(delta_h - hd_target),
        'surface_northing': surface_northing,
        'surface_easting': surface_easting,
        'target_northing': target_northing,
        'target_easting': target_easting,
        'target_depth': target_depth,
        'rkb_elevation': rkb_elevation  # Ini yang paling penting
    })
# Display results if they exist
if st.session_state.results_calculated:
    # Calculate distance to target
    distance_to_target = st.session_state.distance_to_target
    
    # Format the output DataFrame with proper number formatting
    st.header("Trajectory Results Summary")
    
    # Create columns for distance display and copy button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display distance with color coding
        if distance_to_target <= 1.0:
            st.markdown(f"**Distance to target = <span class='distance-good'>{distance_to_target:,.2f} m</span>**", unsafe_allow_html=True)
        else:
            st.markdown(f"**Distance to target = <span class='distance-bad'>{distance_to_target:,.2f} m</span>**", unsafe_allow_html=True)


       
if st.session_state.results_calculated:
    # ... [kode sebelumnya tetap sama sampai display tabel] ...
    
    with col2:
        # Add working copy button for summary table
        csv = st.session_state.summary_df.to_csv(index=False)
        st.download_button(
            "📋 Copy Summary",
            data=csv,
            file_name="trajectory_summary.csv",
            mime="text/csv",
            key="download_summary"
        )
    
    # Display the styled dataframe
    st.dataframe(
        st.session_state.summary_df.style.format({
            'MD': '{:,.2f}',
            'TVD': '{:,.2f}',
            'Inc': '{:,.2f}',
            'Azimuth': '{:,.2f}',
            'N+': '{:,.2f}',
            'E+': '{:,.2f}',
            'Northing': '{:,.2f}',
            'Easting': '{:,.2f}',
            'Displacement': '{:,.2f}',
            'TVDSS': '{:,.2f}',
            'BUR': '{:,.2f}'
        }),
        use_container_width=True
    )
    
# ============== TRAJECTORY VISUALIZATION ==============
    st.header("Trajectory Visualization")
    
    try:
        # Prepare data
        df_viz = st.session_state.detailed_df
        target_disp = sqrt((st.session_state.target_northing - st.session_state.surface_northing)**2 + 
                        (st.session_state.target_easting - st.session_state.surface_easting)**2)
        target_tvd = st.session_state.rkb_elevation - st.session_state.target_depth
        
        # Create two columns with equal width
        col1, col2 = st.columns(2)
        
        # Common width for both plots
        plot_width = 6
        
        with col1:
            # 1. Vertical View Plot (1:2 aspect ratio)
            fig1 = plt.figure(figsize=(plot_width/2, plot_width))  # Width:Height = 2:1
            ax1 = fig1.add_subplot(111)
            
            ax1.plot(df_viz['Displacement'], df_viz['TVD'], 'b-', linewidth=2, label='Trajectory')
            ax1.plot(target_disp, target_tvd, 'rx', markersize=5, label='Target')
            
            # Set limits
            x_padding = max(df_viz['Displacement']) * 0.1
            ax1.set_xlim(-100, max(df_viz['Displacement']) + x_padding)
            
            y_padding = max(df_viz['TVD']) * 0.1
            ax1.set_ylim(max(df_viz['TVD']) + y_padding, min(df_viz['TVD']) - y_padding)
            
            ax1.set_xlabel('Displacement (m)', fontsize=5, fontweight='bold')
            ax1.set_ylabel('TVD (m)', fontsize=5, fontweight='bold')
            ax1.set_title('VERTICAL VIEW', fontsize=6, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.5)
            ax1.legend(loc='upper right', fontsize=4.5)
            
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            # 2. Azimuth View Plot (1:1 aspect ratio) - Centered at (0,0)
            fig2 = plt.figure(figsize=(plot_width, plot_width))
            ax2 = fig2.add_subplot(111)
            
            # Plot relative to surface location (0,0)
            ax2.plot(df_viz['E+'], df_viz['N+'], 'b-', linewidth=2, label='Trajectory')
            
            # Target point relative to surface
            target_eplus = st.session_state.target_easting - st.session_state.surface_easting
            target_nplus = st.session_state.target_northing - st.session_state.surface_northing
            ax2.plot(target_eplus, target_nplus, 'rx', markersize=10, label='Target')
            
            # Calculate boundaries with 20% padding
            max_eplus = max(abs(max(df_viz['E+'])), abs(min(df_viz['E+'])), abs(target_eplus))
            max_nplus = max(abs(max(df_viz['N+'])), abs(min(df_viz['N+'])), abs(target_nplus))
            plot_range = max(max_eplus, max_nplus) * 1.2
            
            # Set axis limits centered at (0,0)
            ax2.set_xlim(-plot_range, plot_range)
            ax2.set_ylim(-plot_range, plot_range)
            ax2.set_aspect('equal')
            
            # Add origin (0,0) marker
            ax2.plot(0, 0, 'ko', markersize=8, label='Surface (0,0)')
            
            # Add axis lines
            ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax2.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            
            ax2.set_xlabel('E+', fontsize=10, fontweight='bold')
            ax2.set_ylabel('N+', fontsize=10, fontweight='bold')
            ax2.set_title('AZIMUTH VIEW', fontsize=12, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.5)
            ax2.legend(loc='best', fontsize=9)
            
            st.pyplot(fig2)
            plt.close()
            
    except Exception as e:
        st.error(f"Error creating plots: {str(e)}")
        
    # Detailed Survey Results
    st.header("Detailed Survey Results")
    
    # Add working copy button for detailed table
    csv = st.session_state.detailed_df.to_csv(index=False)
    st.download_button(
        "📋 Copy Detailed",
        data=csv,
        file_name="detailed_survey.csv",
        mime="text/csv",
        key="download_detailed"
    )
    
    # Display the detailed survey dataframe
    st.dataframe(
        st.session_state.detailed_df.style.format({
            'MD': '{:,.2f}',
            'TVD': '{:,.2f}',
            'Inc': '{:,.2f}',
            'Azimuth': '{:,.2f}',
            'N+': '{:,.2f}',
            'E+': '{:,.2f}',
            'Northing': '{:,.2f}',
            'Easting': '{:,.2f}',
            'Displacement': '{:,.2f}',
            'TVDSS': '{:,.2f}',
            'BUR': '{:,.2f}'
        }),
        use_container_width=True,
        height=min(600, (len(st.session_state.detailed_df) + 1) * 35 + 3)
    )

if st.button("💾 Save to Google Sheets"):
    calculation_data = {
        "well_name": by_well,
        "field_name": by_field,
        "company": by_company,
        "surface_northing": surface_northing,
        "surface_easting": surface_easting,
        "target_northing": target_northing,
        "target_easting": target_easting,
        "target_depth": target_depth,
        "kop": kop,
        "bur": bur if bur else 0,
        "target_inc": target_inc if target_inc else 0,
        "distance_to_target": st.session_state.distance_to_target,
        "summary_df": st.session_state.summary_df,
        "detailed_df": st.session_state.detailed_df
    }
    
    if save_to_gsheets(calculation_data):
        st.success("Data berhasil disimpan ke Google Sheets!")
    else:
        st.error("Gagal menyimpan data")

def show_history():
    st.title("Calculation History")
    
    try:
        # Auth ke Google Sheets
        scope = ["https://spreadsheets.google.com/feeds", 
                "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("gsheets-key.json", scope)
        client = gspread.authorize(creds)
        
        # Baca data
        sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/...").sheet1
        records = sheet.get_all_records()
        
        if not records:
            st.warning("Belum ada data history")
            return
            
        for idx, record in enumerate(records):
            with st.expander(f"{record['well_name']} - {record['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Field:** {record['field_name']}")
                    st.write(f"**KOP:** {record['kop']} m")
                    st.write(f"**BUR:** {record['bur']} °/30m")
                with col2:
                    st.write(f"**Target Inc:** {record['target_inc']}°")
                    st.write(f"**Distance to Target:** {record['distance_to_target']} m")
                
                if st.button("Load Data", key=f"load_{idx}"):
                    # Update session state
                    st.session_state.surface_northing = float(record["surface_northing"])
                    st.session_state.surface_easting = float(record["surface_easting"])
                    st.session_state.target_northing = float(record["target_northing"])
                    st.session_state.target_easting = float(record["target_easting"])
                    st.session_state.target_depth = float(record["target_depth"])
                    st.session_state.kop = float(record["kop"])
                    st.session_state.bur = float(record["bur"])
                    st.session_state.target_inc = float(record["target_inc"])
                    st.success("Data loaded! Kembali ke halaman Calculator")
                    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Tambahkan navigasi di sidebar
page = st.sidebar.radio("Menu", ["Calculator", "History"])
if page == "History":
    show_history()
else:
    pass  # Tampilkan kalkulator utama

              


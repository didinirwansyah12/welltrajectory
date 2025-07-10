import streamlit as st
import pandas as pd
import numpy as np
from math import sin, cos, tan, radians, degrees, sqrt, atan2, acos
import plotly.graph_objects as go
import io

# Helper functions
def calculate_dogleg_angle(inc1, inc2, azi1, azi2):
    inc1_rad = radians(inc1)
    inc2_rad = radians(inc2)
    azi_diff_rad = radians(azi2 - azi1)
    dl_rad = acos(
        cos(inc1_rad) * cos(inc2_rad) + 
        sin(inc1_rad) * sin(inc2_rad) * cos(azi_diff_rad)
    )
    return degrees(dl_rad)

def minimum_curvature(inc1, inc2, azi1, azi2, md):
    dl = radians(calculate_dogleg_angle(inc1, inc2, azi1, azi2))
    if dl < 0.0001:
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
    delta_n = target_northing - surface_northing
    delta_e = target_easting - surface_easting
    delta_h = sqrt(delta_n**2 + delta_e**2)
    azimuth = degrees(atan2(delta_e, delta_n)) % 360
    bur_rad = bur * (np.pi/180) / 30
    radius = 1 / bur_rad
    points = []
    points.append({
        'Parameter': 'RKB', 'MD': 0.0, 'TVD': 0.0, 'Inc': 0.0, 'Azimuth': azimuth,
        'N+': 0.0, 'E+': 0.0, 'Northing': surface_northing, 'Easting': surface_easting,
        'Displacement': 0.0, 'TVDSS': rkb_elevation, 'BUR': 0.0
    })
    points.append({
        'Parameter': 'KOP', 'MD': kop, 'TVD': kop, 'Inc': 0.0, 'Azimuth': azimuth,
        'N+': 0.0, 'E+': 0.0, 'Northing': surface_northing, 'Easting': surface_easting,
        'Displacement': 0.0, 'TVDSS': rkb_elevation - kop, 'BUR': 0.0
    })
    inc = target_inc
    build_length = radius * radians(inc)
    eob_md = kop + build_length
    hd_eob = radius * (1 - cos(radians(inc)))
    tvd_eob = kop + radius * sin(radians(inc))
    n_eob = hd_eob * cos(radians(azimuth))
    e_eob = hd_eob * sin(radians(azimuth))
    points.append({
        'Parameter': 'EOB', 'MD': eob_md, 'TVD': tvd_eob, 'Inc': inc, 'Azimuth': azimuth,
        'N+': n_eob, 'E+': e_eob, 'Northing': surface_northing + n_eob, 'Easting': surface_easting + e_eob,
        'Displacement': hd_eob, 'TVDSS': rkb_elevation - tvd_eob, 'BUR': bur
    })
    delta_tvd_target = (rkb_elevation - target_depth) - tvd_eob
    tangent_length = delta_tvd_target / cos(radians(inc))
    hd_target = hd_eob + tangent_length * sin(radians(inc))
    target_md = eob_md + tangent_length
    n_target = hd_target * cos(radians(azimuth))
    e_target = hd_target * sin(radians(azimuth))
    points.append({
        'Parameter': 'Target', 'MD': target_md, 'TVD': rkb_elevation - target_depth, 'Inc': inc, 'Azimuth': azimuth,
        'N+': n_target, 'E+': e_target, 'Northing': surface_northing + n_target, 'Easting': surface_easting + e_target,
        'Displacement': hd_target, 'TVDSS': target_depth, 'BUR': 0.0
    })
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
        'Parameter': 'TD', 'MD': td_md, 'TVD': tvd_td, 'Inc': inc, 'Azimuth': azimuth,
        'N+': n_td, 'E+': e_td, 'Northing': surface_northing + n_td, 'Easting': surface_easting + e_td,
        'Displacement': hd_td, 'TVDSS': rkb_elevation - tvd_td, 'BUR': 0.0
    })
    detailed_survey = []
    regular_intervals = list(np.arange(0, td_md + 30, 30))
    key_points = [0, kop, eob_md, target_md, td_md]
    all_md_points = sorted(set([round(md, 2) for md in regular_intervals + key_points]))
    for md in all_md_points:
        if md <= kop:
            tvd = md
            inc = 0.0
            hd = 0.0
            n_rel = 0.0
            e_rel = 0.0
            bur_val = 0.0
            parameter = "RKB" if md == 0 else "KOP" if md == kop else "Vertical"
        elif md <= eob_md:
            alpha = (md - kop) / radius
            inc = degrees(alpha)
            tvd = kop + radius * sin(alpha)
            hd = radius * (1 - cos(alpha))
            n_rel = hd * cos(radians(azimuth))
            e_rel = hd * sin(radians(azimuth))
            bur_val = bur
            parameter = "EOB" if md == eob_md else "Build"
        else:
            inc = target_inc
            delta_md = md - eob_md
            tvd = tvd_eob + delta_md * cos(radians(inc))
            hd = hd_eob + delta_md * sin(radians(inc))
            n_rel = hd * cos(radians(azimuth))
            e_rel = hd * sin(radians(azimuth))
            bur_val = 0.0
            parameter = "Target" if md == target_md else ("TD" if md == td_md else "Tangent")
        detailed_survey.append({
            'MD': md, 'TVD': tvd, 'Inc': inc, 'Azimuth': azimuth,
            'N+': n_rel, 'E+': e_rel, 'Northing': surface_northing + n_rel, 'Easting': surface_easting + e_rel,
            'Displacement': hd, 'TVDSS': rkb_elevation - tvd, 'BUR': bur_val, 'Parameter': parameter
        })
    return pd.DataFrame(points), pd.DataFrame(detailed_survey), hd_target, delta_h

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Well Trajectory Planner")
st.markdown("**For J-Type Well with constant Azimuth. Minimum Curvature method.**")

# Session state
if 'results_calculated' not in st.session_state:
    st.session_state.results_calculated = False

# User input
with st.sidebar:
    st.header("Well Design Input")
    surface_northing = st.number_input("Surface Northing (m)", value=9202757.149)
    surface_easting = st.number_input("Surface Easting (m)", value=377233.268)
    ground_level = st.number_input("Ground Level (mASL)", value=1000.00)
    rig_elevation = st.number_input("Rig Elevation (m)", value=0.00)
    rkb_elevation = ground_level + rig_elevation
    st.markdown(f"**RKB Elevation:** {rkb_elevation:.2f} mASL")

    target_northing = st.number_input("Target Northing (m)", value=9202081.409)
    target_easting = st.number_input("Target Easting (m)", value=377153.018)
    target_depth = st.number_input("Target Depth (TVDSS)", value=-1690.74)

    kop = st.number_input("KOP (m)", value=500.0)
    target_inc = st.number_input("Target Inclination (deg)", value=30.0)
    bur = st.number_input("Build-Up Rate (°/30m)", value=2.0)

if st.sidebar.button("Calculate Trajectory"):
    summary_df, detailed_df, hd_target, delta_h = calculate_trajectory(
        surface_northing, surface_easting, rkb_elevation,
        target_northing, target_easting, target_depth,
        kop, bur, target_inc
    )
    st.session_state.results_calculated = True
    st.session_state.summary_df = summary_df
    st.session_state.detailed_df = detailed_df
    st.session_state.distance_to_target = abs(delta_h - hd_target)

# Display results
if st.session_state.results_calculated:
    st.subheader("Summary Table")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.dataframe(st.session_state.summary_df, use_container_width=True)
    with col2:
        csv = st.session_state.summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Summary CSV",
            data=csv,
            file_name='summary_table.csv',
            mime='text/csv'
        )

    st.subheader("Detailed Survey")
    col3, col4 = st.columns([4, 1])
    with col3:
        st.dataframe(st.session_state.detailed_df, use_container_width=True, height=500)
    with col4:
        csv2 = st.session_state.detailed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Detailed Survey CSV",
            data=csv2,
            file_name='detailed_survey.csv',
            mime='text/csv'
        )

    st.subheader("Trajectory Plots")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=st.session_state.detailed_df['Easting'],
        y=st.session_state.detailed_df['Northing'],
        mode='lines+markers',
        name='Plan View'
    ))
    fig1.update_layout(title="Plan View (Top View)", xaxis_title="Easting (m)", yaxis_title="Northing (m)", height=500)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=st.session_state.detailed_df['MD'],
        y=st.session_state.detailed_df['TVDSS'],
        mode='lines+markers',
        name='Profile View'
    ))
    fig2.update_layout(title="Profile View (MD vs TVDSS)", xaxis_title="Measured Depth (m)", yaxis_title="TVDSS (m)", height=500, yaxis_autorange='reversed')
    st.plotly_chart(fig2, use_container_width=True)

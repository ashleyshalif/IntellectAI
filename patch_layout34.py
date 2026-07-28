with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

changes = []

old_settings = '''with st.sidebar:
    st.markdown("### Inspection Settings")
    category = st.selectbox("Product category", CATEGORIES)

    if "last_category" not in st.session_state or st.session_state.last_category != category:
        cal = CALIBRATED_THRESHOLDS.get(category, {"minor": 7.0, "defect": 12.0})
        st.session_state.threshold_slider = cal["defect"]
        st.session_state.minor_threshold_slider = cal["minor"]
        st.session_state.last_category = category

    st.markdown('<div class="sidebar-subhead">Thresholds</div>', unsafe_allow_html=True)
    threshold = st.slider("Anomaly threshold", 0.0, 30.0, 0.5,
                           key="threshold_slider",
                           help="Max Mahalanobis distance above which a part is flagged defective")
    minor_threshold = st.slider("Minor irregularity threshold", 0.0, 30.0, 0.5,
                           key="minor_threshold_slider",
                           help="Scores above this but below the defect threshold are flagged as minor irregularities (e.g. glue strips, oil marks) instead of full defects")
    st.caption(f"Calibrated from real test-set scores for **{category}** "
               f"(minor={CALIBRATED_THRESHOLDS.get(category, {}).get('minor', 'n/a')}, "
               f"defect={CALIBRATED_THRESHOLDS.get(category, {}).get('defect', 'n/a')}). "
               f"Adjust manually if needed.")

stats = get_stats(category)'''

new_settings = '''with st.container(border=True):
    st.markdown('<div class="sidebar-subhead">Inspection settings</div>', unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        category = st.selectbox("Product category", CATEGORIES)

    if "last_category" not in st.session_state or st.session_state.last_category != category:
        cal = CALIBRATED_THRESHOLDS.get(category, {"minor": 7.0, "defect": 12.0})
        st.session_state.threshold_slider = cal["defect"]
        st.session_state.minor_threshold_slider = cal["minor"]
        st.session_state.last_category = category

    with ctrl2:
        threshold = st.slider("Anomaly threshold", 0.0, 30.0, 0.5,
                               key="threshold_slider",
                               help="Max Mahalanobis distance above which a part is flagged defective")
    with ctrl3:
        minor_threshold = st.slider("Minor irregularity threshold", 0.0, 30.0, 0.5,
                               key="minor_threshold_slider",
                               help="Scores above this but below the defect threshold are flagged as minor irregularities (e.g. glue strips, oil marks) instead of full defects")

    st.caption(f"Calibrated from real test-set scores for **{category}** "
               f"(minor={CALIBRATED_THRESHOLDS.get(category, {}).get('minor', 'n/a')}, "
               f"defect={CALIBRATED_THRESHOLDS.get(category, {}).get('defect', 'n/a')}). "
               f"Adjust manually if needed.")

    st.markdown('<div class="sidebar-subhead">Diagnostics</div>', unsafe_allow_html=True)
    with st.expander("Model Performance"):
        st.caption("Offline benchmark from evaluate.py against real MVTec test-set "
                    "images (top1pct scorer). Not a live per-image confidence score.")
        BENCHMARK = {
            "bottle": {"broken_large": "20/20", "broken_small": "22/22", "contamination": "19/21"},
            "cable": {"bent_wire": "5/13", "cable_swap": "0/12", "combined": "3/11",
                      "cut_inner_insulation": "3/14", "cut_outer_insulation": "5/10",
                      "missing_cable": "0/12", "missing_wire": "1/10", "poke_insulation": "1/10"},
            "tile": {"crack": "16/17", "glue_strip": "18/18", "gray_stroke": "3/16",
                     "oil": "18/18", "rough": "11/15"},
        }
        for defect, ratio in BENCHMARK.get(category, {}).items():
            st.write(f"**{defect}**: {ratio} detected in test set")

        st.markdown("---")
        rate = get_agreement_rate(category)
        if rate:
            agree, total = rate
            st.metric(f"User agreement ({category})", f"{agree}/{total}", f"{100*agree/total:.0f}%")
        else:
            st.caption("No user feedback logged yet for this category.")

stats = get_stats(category)'''

if old_settings in content:
    content = content.replace(old_settings, new_settings)
    changes.append("settings moved from sidebar into a top bordered control bar (category + both thresholds side by side), Diagnostics panel moved up here too")
else:
    print("WARNING: settings sidebar block not found")

old_diag_old_location = '''with st.sidebar:
    st.markdown('<div class="sidebar-subhead">Diagnostics</div>', unsafe_allow_html=True)
    with st.expander("Model Performance"):
        st.caption("Offline benchmark from evaluate.py against real MVTec test-set "
                    "images (top1pct scorer). Not a live per-image confidence score.")
        BENCHMARK = {
            "bottle": {"broken_large": "20/20", "broken_small": "22/22", "contamination": "19/21"},
            "cable": {"bent_wire": "5/13", "cable_swap": "0/12", "combined": "3/11",
                      "cut_inner_insulation": "3/14", "cut_outer_insulation": "5/10",
                      "missing_cable": "0/12", "missing_wire": "1/10", "poke_insulation": "1/10"},
            "tile": {"crack": "16/17", "glue_strip": "18/18", "gray_stroke": "3/16",
                     "oil": "18/18", "rough": "11/15"},
        }
        for defect, ratio in BENCHMARK.get(category, {}).items():
            st.write(f"**{defect}**: {ratio} detected in test set")

        st.markdown("---")
        rate = get_agreement_rate(category)
        if rate:
            agree, total = rate
            st.metric(f"User agreement ({category})", f"{agree}/{total}", f"{100*agree/total:.0f}%")
        else:
            st.caption("No user feedback logged yet for this category.")

st.markdown("---")'''

new_diag_old_location = '''st.markdown("---")'''

if old_diag_old_location in content:
    content = content.replace(old_diag_old_location, new_diag_old_location)
    changes.append("removed the old duplicate Diagnostics block from the bottom of the page (now lives at the top only)")
else:
    print("WARNING: old bottom Diagnostics block not found")

old_layout = '''                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original**")
                    st.image(r["pil_img"], width=320)
                with col2:
                    st.markdown("**Anomaly Heatmap**")
                    st.image(r["blended"], width=320)

                st.write("")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""<div class="metric-card"><span class="metric-index">01</span><div class="metric-value {r['css_class']}">{r['status']}</div>
                                <div class="metric-label">Inspection Result</div></div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-card"><span class="metric-index">02</span><div class="metric-value">{r['anomaly_score']:.2f}</div>
                                <div class="metric-label">Anomaly Score</div></div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="metric-card"><span class="metric-index">03</span><div class="metric-value">{r['threshold']:.1f}</div>
                                <div class="metric-label">Threshold</div></div>""", unsafe_allow_html=True)'''

new_layout = '''                img_col1, img_col2, metrics_col = st.columns([2, 2, 1])
                with img_col1:
                    st.markdown("**Original**")
                    st.image(r["pil_img"], width=280)
                with img_col2:
                    st.markdown("**Anomaly Heatmap**")
                    st.image(r["blended"], width=280)
                with metrics_col:
                    st.markdown(f"""<div class="metric-card"><span class="metric-index">01</span><div class="metric-value {r['css_class']}">{r['status']}</div>
                                <div class="metric-label">Inspection Result</div></div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="metric-card"><span class="metric-index">02</span><div class="metric-value">{r['anomaly_score']:.2f}</div>
                                <div class="metric-label">Anomaly Score</div></div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="metric-card"><span class="metric-index">03</span><div class="metric-value">{r['threshold']:.1f}</div>
                                <div class="metric-label">Threshold</div></div>""", unsafe_allow_html=True)'''

if old_layout in content:
    content = content.replace(old_layout, new_layout)
    changes.append("images widened into two columns, metric cards now a stacked side rail next to them instead of a row below")
else:
    print("WARNING: image/metrics layout block not found")

old_card_css = '''    .metric-card {
        background: var(--bg-panel); border: 1px dashed var(--border); border-radius: 8px;
        padding: 20px 22px; text-align: left; position: relative;
        transition: transform 0.32s cubic-bezier(0.22, 1, 0.36, 1),
                    border-color 0.32s cubic-bezier(0.22, 1, 0.36, 1),
                    box-shadow 0.32s cubic-bezier(0.22, 1, 0.36, 1);
    }'''

new_card_css = '''    .metric-card {
        background: var(--bg-panel); border: 1px dashed var(--border); border-radius: 8px;
        padding: 20px 22px; margin-bottom: 12px; text-align: left; position: relative;
        transition: transform 0.32s cubic-bezier(0.22, 1, 0.36, 1),
                    border-color 0.32s cubic-bezier(0.22, 1, 0.36, 1),
                    box-shadow 0.32s cubic-bezier(0.22, 1, 0.36, 1);
    }'''

if old_card_css in content:
    content = content.replace(old_card_css, new_card_css)
    changes.append("added spacing between stacked metric cards in the new rail")
else:
    print("WARNING: metric-card CSS block not found")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Applied:", ", ".join(changes) if changes else "NOTHING -- check warnings above")
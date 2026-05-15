if st.session_state.stage == "splash":
    st.markdown("""
    <div class="splash-wrap">
      <div class="splash-icon">🚗</div>
      <div class="splash-title">ParkVision Pro</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Start Parking Space Detection"):
        st.session_state.stage = "upload"
        st.rerun()

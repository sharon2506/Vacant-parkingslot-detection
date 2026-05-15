elif st.session_state.stage == "draw":

    frame = st.session_state.first_frame

    st.components.v1.html(full_html, height=800)

    mj = st.text_area(
        "",
        height=60,
        placeholder='[{"x1":100,"y1":50}]'
    )

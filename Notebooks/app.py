# app.py

import collaborative_filtering
import about
import streamlit as st

PAGES = {
    "Recommender": collaborative_filtering,
    "About": about
}

#st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

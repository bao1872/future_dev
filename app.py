import streamlit as st

st.set_page_config(
    page_title="future_dev · Strategy Research",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1800px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #263440;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = [
    st.Page(
        "pages/2_Chart.py",
        title="研究工作台",
        default=True,
    ),
    st.Page(
        "pages/3_Strategy_Lab.py",
        title="策略实验",
    ),
    st.Page(
        "pages/4_Results.py",
        title="实验对比",
    ),
    st.Page(
        "pages/1_Data.py",
        title="数据状态",
    ),
]

pg = st.navigation(pages)
pg.run()

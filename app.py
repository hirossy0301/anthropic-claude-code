"""Streamlit 画面: streamlit run app.py

上部のメニューで画面を切り替える。
"""
import streamlit as st

st.set_page_config(page_title="競艇 統計予想", layout="wide")
page = st.navigation([
    st.Page("views/predict.py", title="予想", icon=":material/sports_score:", default=True),
    st.Page("views/paper.py", title="EV記録", icon=":material/receipt_long:"),
    st.Page("views/simulation.py", title="シミュレーション", icon=":material/show_chart:"),
    st.Page("views/method.py", title="予測の仕組み", icon=":material/menu_book:"),
], position="top")
page.run()

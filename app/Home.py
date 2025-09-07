import streamlit as st
from app_text import home_page_text_intro


def main():
    st.set_page_config(
        page_title="Big Data Bowl Play Model Animator",
        page_icon="🏈",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.markdown("## Big Data Bowl Historic Model Animator 🏈")
    st.image("https://media.giphy.com/media/cQBhRosP3tzIyCfIHj/giphy.gif")
    st.divider()
    st.markdown(home_page_text_intro)

    # st.divider()
    # st.markdown(app_overview)


if __name__ == "__main__":
    main()

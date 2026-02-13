import streamlit as st

st.set_page_config(page_title="Test", layout="wide")
st.title("Dashboard is alive")
st.write("If you see this, Streamlit Cloud is working.")

# Test imports one by one
errors = []

try:
    import pandas as pd
    st.success(f"pandas {pd.__version__}")
except Exception as e:
    errors.append(f"pandas: {e}")

try:
    import numpy as np
    st.success(f"numpy {np.__version__}")
except Exception as e:
    errors.append(f"numpy: {e}")

try:
    import plotly
    st.success(f"plotly {plotly.__version__}")
except Exception as e:
    errors.append(f"plotly: {e}")

try:
    import yfinance as yf
    st.success(f"yfinance {yf.__version__}")
except Exception as e:
    errors.append(f"yfinance: {e}")

try:
    import nltk
    st.success(f"nltk {nltk.__version__}")
except Exception as e:
    errors.append(f"nltk: {e}")

try:
    from textblob import TextBlob
    st.success("textblob OK")
except Exception as e:
    errors.append(f"textblob: {e}")

if errors:
    st.error("Failed imports:")
    for err in errors:
        st.code(err)
else:
    st.balloons()
    st.success("All imports passed!")

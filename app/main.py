import streamlit as st
import pandas as pd

# Set the page title and header
st.set_page_config(page_title="My First App", page_icon="🚀")
st.title("🚀 My First Streamlit App")
st.write("Welcome! This is a basic Streamlit application.")

# 1. Text Input
name = st.text_input("What is your name?")
if name:
    st.write(f"Hello, **{name}**! 👋")

# 2. Slider
age = st.slider("How old are you?", 0, 100, 25)
st.write(f"You are {age} years old.")

# 3. Checkbox
if st.checkbox("Show a secret message"):
    st.success("You found the secret message! 🎉")

# 4. Displaying Data
st.subheader("Sample Data Table")
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Tokyo']
})
st.dataframe(data)
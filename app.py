from dotenv import load_dotenv
load_dotenv() ## LOADING ALL THE ENVIROMENT VARIABLES

import streamlit as st
import os
import sqlite3

import google.generativeai as genai

## Configure our API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to Load Google Gemini Model & Provide SQL Query as response
def get_gemini_response(question, prompt):
  model = genai.GenerativeModel('gemini-pro')
  response=model.generate_content([prompt,question])
  return response.txt

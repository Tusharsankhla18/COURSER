
import streamlit as st
import pickle
import streamlit.components.v1 as staticmethod

# load EDA
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# Loading our dataset
def load_data(data):
  df = pd.read_csv(data)
  return df


# func.
# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
  count_vect = CountVectorizer()
  cv_mat = count_vect.fit_transform(data)


  # Get cosine
  cosine_sim_mat = cosine_similarity(cv_mat)
  return cosine_sim_mat

# Recommendation System
@st.cache_data
def get_recommendation(title,cosine_sim_mat,df,num_of_rec =5):

  # indices of the course
  course_indices = pd.Series(df.index,index = df['course_title']).drop_duplicates()
  # Index of the course
  idx = course_indices[title]

  # look into the cosine matrix for that index
  sim_scores = list(enumerate(cosine_sim_mat[idx]))
  sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse =True)
  selected_course_indices = [i[0] for i in sim_scores[1:]]
  selected_course_courses = [i[0] for i in sim_scores[1:]]

  # Get dataframe and title
  result_df = df.iloc[selected_course_indices]
  result_df['similarity_scores']= selected_course_scores
  final_recommendation = result_df[['course_title', 'similarity_scores', 'url', 'price', 'num_subscribers']]
  return final_recommendation.head(num_of_rec)


# Search for Course
@st.cache_data
def search_term_if_not_found(term,df):
  result_df = df[df['course_title'].str.contains(term)]
  return result_df

# Doing some CSS Style to make it more presentable

RESULT_TEMP = """
<div style = "width:90%; height:100%; margin:1px; padding:5px; position:relative; border-radius:5px; border-bottom;
box-shadow:0 0 15px 5px #ccc; background-color; #a8f0c6;
border-left:5px solid #6c6c6c;">
<h4>{}</h4>
<p style = "color:blue;"><span style = "color:black;">Score:</span>{}</p>
<p style = "color:blue;"><span style = "color:black;">URL:</span><a href = "{}", target ="_blank">Link</a></p>
<p style = "color:blue;"><span style = "color:black;">Price:</span>{}</p>
<p style = "color:blue;"><span style = "color:black;">Number of Subscribers:</span>{}</p>

</div>
"""



def recommendation():
  st.title("COURSER")
  menu = ["Home", "Recommend", "About"]
  choice  = st.sidebar.selectbox("Menu",menu)

  df = load_data("udemy_courses_data.csv")
  if choice == "Home":
    st.subheader("Home")
    st.dataframe(df.head(10))


  elif choice =="Recommend":
    st.subheader("The one-stop solution for you to find courses.")
    cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
    search_term = st.text_input("Search")
    num_of_rec = st.sidebar.number_input("Number",4,30,7)
    if st.button("Recommend"):
      if search_term is not None:
        try:
          result = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
          for row in result.iterrows():
            rec_title=row[1][0]
            rec_score=row[1][1]
            rec_url=row[1][2]
            rec_price=row[1][3]
            rec_sub=row[1][4]

            #st.write("Title : ", rec_title)
            stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_url,rec_price,rec_sub),height=350)

        except:
          result = "OOPs, Not Found"
          st.warning(result)
          st.info("Suggested Options Include")
          result_df = search_term_if_not_found(search_term)
          st.dataframe(result_df)


    else:
      st.subheader("About")
      st.text("Built with Streamlit & Pandas")

recommendation()



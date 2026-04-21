import pickle as pkl
import numpy as np
import streamlit as st
import pandas as pd

st.title("Book Recommendation System")
model = pkl.load(open('model_book.pkl', 'rb'))
books = pd.read_csv('final_books.csv')
book_pivot= pd.read_csv('book_pivot.csv',index_col=0)
selected_book = st.selectbox("Enter the Book Name", book_pivot.index.sort_values().unique())
print(np.where(book_pivot.index=='Harry Potter and the Goblet of Fire (Book 4)'))

def change(value):
    list = value.split('http')
    list[0]='https'
    return "".join(list)
books['poster'] =books['poster'].apply(change)
def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1),
        n_neighbors=6
    )
    recommended_books = []
    for i in suggestions[0]:
        recommended_books.append(book_pivot.index[i])
    return recommended_books[1:]

print(books)
if st.button("Recommend"):
    st.header("Your book journey begins from here :) ")
    recommendations = recommend_book(selected_book)
    cols = st.columns(5)
    for i , book in enumerate(recommendations):
        poster=books[books['name'] == book]['poster'].values[0]
        with cols[i]:
            st.image(poster)
            st.caption(book,text_alignment='left')

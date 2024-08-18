import pandas as pd
import pickle
import streamlit as st
from fuzzywuzzy import process
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your data and TF-IDF matrix from pickle files
with open(r'C:\Users\PJ COMPUTERS\Desktop\BOOK_Full\.ipynb_checkpoints\df.pkl', 'rb') as file:
    data = pickle.load(file)

# Ensure 'Edition_author' column is filled properly
data['Edition_author'] = data['Edition_author'].fillna('')

# Initialize the TF-IDF vectorizer and fit-transform on the 'Edition_author' column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Edition_author'])

# Function for fuzzy recommendations
def get_fuzzy_recommendations(title, tfidf_matrix, data, top_n=6):
    closest_match = process.extractOne(title, data['Book-Title'])

    if closest_match is None or closest_match[1] == 0:
        return data.sample(n=min(top_n, len(data)))[['Book-Title', 'Book-Author', 'Ratings', 'Reviews', 'Image-URL-M']]
    
    match_title = closest_match[0]
    match_score = closest_match[1]

    matching_indices = data[data['Book-Title'].str.strip() == match_title.strip()].index

    if matching_indices.empty:
        st.write(f"No index found for the match '{match_title}'. Providing fallback recommendations.")
        return data.sample(n=min(top_n, len(data)))[['Book-Title', 'Book-Author', 'Ratings', 'Reviews', 'Image-URL-M']]

    idx = matching_indices[0]

    if tfidf_matrix[idx:idx+1].shape[0] == 0:
        st.write(f"TF-IDF matrix is empty for the matched title '{match_title}'. Providing fallback recommendations.")
        return data.sample(n=min(top_n, len(data)))[['Book-Title', 'Book-Author', 'Ratings', 'Reviews', 'Image-URL-M']]

    sim_scores = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][1:top_n + 1]
    top_books = data.iloc[top_indices][['Book-Title', 'Book-Author', 'Ratings', 'Reviews', 'Image-URL-M']]

    return top_books

# Function to display the book recommendations
def recommendations_page():
    st.title("📚 Book Recommendation System")
    st.write("Enter a book title to get similar book recommendations.")
    
    title = st.text_input("Enter a book title:", placeholder="e.g., The Great Gatsby")
    
    if st.button("Get Recommendations"):
        with st.spinner("Processing..."):
            if title.strip():
                recommendations = get_fuzzy_recommendations(title, tfidf_matrix, data)
                if not recommendations.empty:
                    st.write("### Top Recommended Books:")
                    
                    # Create two columns for the layout
                    col1, col2 = st.columns(2)
                    
                    # Split the recommendations into two lists for the two rows
                    first_row = recommendations.iloc[:3]
                    second_row = recommendations.iloc[3:]
                    
                    # Function to add a dark horizontal line
                    def dark_line():
                        st.markdown("<hr style='border: 3px solid white;'>", unsafe_allow_html=True)
                    
                    with col1:
                        for idx, row in first_row.iterrows():
                            st.image(row['Image-URL-M'], width=150, caption=row['Book-Title'])
                            st.write(f"**Title:** {row['Book-Title']}")
                            st.write(f"**Author:** {row['Book-Author']}")
                            st.write(f"**Ratings:** {row['Ratings']:.1f} {rating_to_stars(row['Ratings'])}")
                            st.write(f"**Reviews:** {int(row['Reviews'])}")
                            dark_line()  # Add dark horizontal line
                    
                    with col2:
                        for idx, row in second_row.iterrows():
                            st.image(row['Image-URL-M'], width=150, caption=row['Book-Title'])
                            st.write(f"**Title:** {row['Book-Title']}")
                            st.write(f"**Author:** {row['Book-Author']}")
                            st.write(f"**Ratings:** {row['Ratings']:.1f} {rating_to_stars(row['Ratings'])}")
                            st.write(f"**Reviews:** {int(row['Reviews'])}")
                            dark_line()  # Add dark horizontal line
                else:
                    st.write("No recommendations found.")
            else:
                st.write("Please enter a book title.")

# Function to display the login page
def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username and password:
            # In a real app, you would validate the username and password here
            st.session_state['logged_in'] = True
            st.success("Logged in successfully!")
        else:
            st.error("Please enter both username and password.")

# Function to display the home page
def home_page():
    st.title("Home Page")
    st.write("Welcome to the Book Recommendation System! Navigate through the sidebar to explore different pages.")

    filtered_books = data[(data['Reviews'] > 100) & (data['Ratings'] >4)]

    st.write("Enjoy your learning!!!")
    for idx, row in filtered_books.iterrows():
        st.image(row['Image-URL-M'], width=150, caption=row['Book-Title'])
        st.write(f"**Title:** {row['Book-Title']}")
        st.write(f"**Author:** {row['Book-Author']}")
        st.write(f"**Ratings:** {row['Ratings']:.1f} {rating_to_stars(row['Ratings'])}")
        st.write(f"**Reviews:** {int(row['Reviews'])}")

         #Add a white broad line to separate books
        st.markdown("<hr style='border: 3px solid white;'>", unsafe_allow_html=True)
# Function to convert rating to star icons
def rating_to_stars(rating):
    full_stars = int(rating)
    half_star = 1 if rating - full_stars >= 0.5 else 0
    empty_stars = 5 - (full_stars + half_star)
    
    star_icons = "⭐" * full_stars + "⭐" * half_star + "☆" * empty_stars
    return star_icons

# Main function to manage the app's pages
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Login", "Home", "Book Recommendations"])
    
    # Check if user is logged in
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if page == "Login":
        if st.session_state['logged_in']:
            st.sidebar.write("You are already logged in.")
        else:
            login_page()
    elif page == "Home":
        home_page()
    elif page == "Book Recommendations":
        if st.session_state['logged_in']:
            recommendations_page()
        else:
            st.write("Please log in to access the Book Recommendations page.")

if __name__ == "__main__":
    main()
# Smart Movie Recommendation System 🎬

A robust, content-based movie recommendation engine built with Python, Streamlit, Pandas, and Scikit-learn. This application allows users to discover new movies by searching intuitively across movie titles, actors, directors, or genres. 

## Features 🚀

- **Intelligent Search Engine**: The system automatically detects whether your query is a movie title, an actor, a director, or a genre and adapts accordingly.
- **Content-Based Filtering**: Leverages TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and Cosine Similarity to find the most relevant recommendations based on cast, crew, genres, and movie overviews.
- **Modern UI**: A fully responsive, dark-themed user interface built with custom CSS using Streamlit. Includes hover-effects, dynamic tags, and intuitive layouts.
- **Dynamic Results**: Instantly view highly-rated movies filtering by your search query, or get recommended titles if you query a specific movie.
- **Customizable**: Adjust the number of recommendations returned via an interactive slider.

## Tech Stack 🛠️

- **Frontend**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Styling**: HTML/CSS embedded in Streamlit

## Setup & Installation ⚙️

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vishnucax/smart-movie-recommendation-system.git
   cd smart-movie-recommendation-system
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Data Preprocessing:**
   (Note: Ensure your raw dataset files are placed inside the `raw datasets` directory if you plan to re-process the data)
   ```bash
   python preprocess.py
   ```
   *This will generate the `final_movies.csv` required by the application.*

5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Usage 💡

1. Open the application in your browser (usually `http://localhost:8501`).
2. Use the search bar to enter a query. This could be:
   - A Movie Title (e.g., *The Matrix*)
   - An Actor (e.g., *Brad Pitt*)
   - A Director (e.g., *Christopher Nolan*)
   - A Genre (e.g., *Action*, *Sci-Fi*)
3. Adjust the limit slider to choose how many results/recommendations you want to see.
4. Hit **Search Engine** to fetch and view your results.

## Contributing 🤝

Contributions are welcome! Please fork the repository and create a pull request for any improvements or bug fixes.

---
*Developed with ❤️*

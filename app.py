import streamlit as st
import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Intelligent Product Search", layout="wide")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------- Session ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = ""

if "search_history" not in st.session_state:
    st.session_state.search_history = []

# ---------------- Registration ----------------
def register():

    st.title("📝 User Registration")

    new_user = st.text_input("Create Username")
    new_pass = st.text_input("Create Password", type="password")

    if st.button("Register"):

        if os.path.exists("users.csv"):
            users = pd.read_csv("users.csv")
        else:
            users = pd.DataFrame(columns=["username","password","role"])

        if new_user in users["username"].values:
            st.error("Username already exists")

        else:

            new_row = pd.DataFrame([{
                "username": new_user,
                "password": new_pass,
                "role": "user"
            }])

            users = pd.concat([users, new_row], ignore_index=True)
            users.to_csv("users.csv", index=False)

            st.success("Registration successful! Please login.")
# ---------------- Login ----------------
def login():

    st.title("🔐 Login System")

    users = pd.read_csv("users.csv")

    username = st.text_input("Username").strip()
    password = st.text_input("Password", type="password").strip()


    if st.button("Login"):

        user = users[
         (users["username"].astype(str).str.strip() == username) &
         (users["password"].astype(str).str.strip() == password)
]
        if not user.empty:

            st.session_state.logged_in = True
            st.session_state.role = user.iloc[0]["role"]

            st.rerun()

        else:
            st.error("Invalid Credentials")

# ---------------- Login/Register Switch ----------------
if not st.session_state.logged_in:

    page = st.sidebar.selectbox("Select Page", ["Login", "Register"])

    if page == "Login":
        login()
    else:
        register()

    st.stop()

# ---------------- Logout ----------------
st.sidebar.write(f"Logged in as: {st.session_state.role}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.role = ""
    st.rerun()

# ---------------- Load Data ----------------
data = pd.read_csv("products.csv")

# Ensure rating columns exist
if "rating_sum" not in data.columns:
    data["rating_sum"] = 0

if "rating_count" not in data.columns:
    data["rating_count"] = 0

# Fix missing columns
if "image" not in data.columns:
    data["image"] = ""

data["name"] = data["name"].fillna("Unknown Product")

data["text"] = (
    data["name"].fillna("") + " " +
    data["description"].fillna("") + " " +
    data["category"].fillna("")
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["text"])

st.title("🧠 Intelligent Product Search Engine")


# ---------------- Admin Panel ----------------
if st.session_state.role == "admin":

    st.sidebar.header("Admin Panel")

    # ------- Add Product -------
    st.sidebar.subheader("Add Product")

    name = st.sidebar.text_input("Product Name")
    category = st.sidebar.text_input("Category")
    price = st.sidebar.number_input("Price", min_value=0)
    color = st.sidebar.text_input("Color")
    brand = st.sidebar.text_input("Brand")
    description = st.sidebar.text_area("Description")

    image_file = st.sidebar.file_uploader(
        "Upload Product Image", type=["png", "jpg", "jpeg"]
    )

    if st.sidebar.button("Add Product"):

        image_path = ""

        if image_file is not None:

            if not os.path.exists("images"):
                os.makedirs("images")

            image_path = os.path.join("images", image_file.name)

            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())

        new_product = pd.DataFrame([{
            "name": name,
            "category": category,
            "price": price,
            "color": color,
            "brand": brand,
            "description": description,
            "rating_sum": 0,
            "rating_count": 0,
            "image": image_path
        }])

        data = pd.concat([data, new_product], ignore_index=True)
        data.to_csv("products.csv", index=False)

        st.sidebar.success("Product Added Successfully")

    # ------- Delete Product -------
    st.sidebar.subheader("Delete Product")

    product_to_delete = st.sidebar.selectbox(
        "Select Product",
        data["name"]
    )

    if st.sidebar.button("Delete Product"):

        data = data[data["name"] != product_to_delete]
        data.to_csv("products.csv", index=False)

        st.sidebar.success("Product Deleted Successfully")
# ---------------- Filters ----------------
st.sidebar.header("Filters")

min_price = int(data["price"].min())
max_price = int(data["price"].max())

selected_price = st.sidebar.slider(
    "Price Range", min_price, max_price, (min_price, max_price)
)

selected_category = st.sidebar.selectbox(
    "Category", ["All"] + list(data["category"].unique())
)

selected_color = st.sidebar.selectbox(
    "Color", ["All"] + list(data["color"].unique())
)

sort_option = st.sidebar.selectbox(
    "Sort By Price", ["None", "Low to High", "High to Low"]
)

query = st.text_input("Search Product")

# ---------------- Clean Query ----------------
def clean_query(query):

    words = query.lower().split()

    filtered = [word for word in words if word not in stop_words]

    return " ".join(filtered)

# ---------------- Recommendation ----------------
def get_recommendations(selected_row, data):

    category = selected_row["category"]
    price = selected_row["price"]

    recs = data[
        (data["category"] == category) &
        (data["price"].between(price - 2000, price + 2000))
    ]

    return recs.head(3)

# ---------------- Apply Filters ----------------
results = data.copy()

results = results[
    (results["price"] >= selected_price[0]) &
    (results["price"] <= selected_price[1])
]

if selected_category != "All":
    results = results[results["category"] == selected_category]

if selected_color != "All":
    results = results[results["color"] == selected_color]

# ---------------- AI Search ----------------
if query:

    cleaned_query = clean_query(query)

    st.session_state.search_history.append(
        (cleaned_query, datetime.now())
    )

    query_vector = vectorizer.transform([cleaned_query])

    similarity = cosine_similarity(query_vector, tfidf_matrix)

    similarity_scores = similarity.flatten()

    top_indices = similarity_scores.argsort()[::-1][:10]

    results = data.iloc[top_indices].copy()

    results["relevance"] = similarity_scores[top_indices]

# ---------------- Sorting ----------------
if sort_option == "Low to High":
    results = results.sort_values(by="price")

elif sort_option == "High to Low":
    results = results.sort_values(by="price", ascending=False)

# ---------------- Display ----------------
st.subheader(f"Results Found: {len(results)}")

if not results.empty:

    for i, row in results.iterrows():

        st.markdown("---")

        st.subheader(row["name"])

        if pd.notna(row["image"]) and row["image"] != "":
            st.image(row["image"], width=200)

        st.write("Category:", row["category"])
        st.write("Price: ₹", row["price"])
        st.write("Color:", row["color"])
        st.write("Brand:", row["brand"])
        st.write("Description:", row["description"])

        if "relevance" in row:
            st.write("Relevance Score:", round(row["relevance"] * 100, 2), "%")

        if row["rating_count"] > 0:
            avg_rating = round(row["rating_sum"] / row["rating_count"], 1)
            st.write(f"⭐ Average Rating: {avg_rating}")
        else:
            st.write("⭐ No ratings yet")

        rating = st.slider(f"Rate {row['name']}", 1, 5, key=f"slider_{i}")

        if st.button(f"Submit Rating for {row['name']}", key=f"btn_{i}"):

            data.loc[data["name"] == row["name"], "rating_sum"] += rating
            data.loc[data["name"] == row["name"], "rating_count"] += 1

            data.to_csv("products.csv", index=False)

            st.success("Rating Submitted Successfully")

        st.markdown("### 🔎 You May Also Like")

        recs = get_recommendations(row, data)

        for _, rec in recs.iterrows():

            if rec["name"] != row["name"]:
                st.write("👉", rec["name"], "- ₹", rec["price"])

else:
    st.write("No Products Found")

# ---------------- Search History ----------------
st.sidebar.header("Search History")

for item in st.session_state.search_history[-5:]:
    st.sidebar.write(item[0])
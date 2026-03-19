import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsTransformer

def load_data_files():
    books = pd.read_csv("../data/Books.csv")
    ratings = pd.read_csv("../data/Ratings.csv")
    books.drop_duplicates("Book-Title", inplace=True)
    merged = ratings.merge(books, on="ISBN")
    merged.drop(["ISBN", "Image-URL-S", "Image-URL-M", "Image-URL-L", "Publisher"], axis=1, inplace=True)
    merged.dropna(inplace=True)
    return books, merged


def extract_features(raw_table):
    x = raw_table.groupby("User-ID").count()["Book-Rating"] > 100
    expert_users = x[x].index

    filtered_ratings = raw_table[raw_table["User-ID"].isin(expert_users)]

    y = filtered_ratings.groupby("Book-Title").count()["Book-Rating"] >= 50
    famous_books = y[y].index

    user_ratings = filtered_ratings[filtered_ratings["Book-Title"].isin(famous_books)]

    design_matrix = user_ratings.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating")
    design_matrix.fillna(0, inplace=True)
    return design_matrix

def make_model(matrix):
    scaler = StandardScaler()
    model = KNeighborsTransformer(n_neighbors=5, mode="distance", metric="cosine")
    scaled = scaler.fit_transform(matrix)
    transform = model.fit_transform(scaled)
    return transform

def recommend(book_name, table, design_matrix, model):
    book_index = np.where(design_matrix.index==book_name)[0][0]
    csr = model[book_index]
    for row in range(csr.shape[0]):
        start_index = csr.indptr[row] + 1
        end_index = csr.indptr[row + 1]
        for index in range(start_index, end_index):
            value = csr.data[index]
            column= csr.indices[index]
            print(f"{table[table["Book-Title"]==design_matrix.index[column]][["Book-Title", "Book-Author"]].values.flatten()}")
   

def main(name):
    books, table = load_data_files()
    matrix = extract_features(table)
    model = make_model(matrix)
    recommend(name, books, matrix, model)

if __name__ == '__main__':
    name = input("Input a book name: ")
    main(name)
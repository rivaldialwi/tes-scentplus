import pandas as pd
import streamlit as st
import joblib
import nltk
import sqlite3
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder

# Mengunduh resource stopwords jika belum ada
nltk.download('stopwords')
nltk.download('punkt')

# Membaca model yang sudah dilatih dan TF-IDF Vectorizer
@st.cache(allow_output_mutation=True)
def load_model_and_vectorizer():
    model = joblib.load("model100.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

logreg_model, tfidf_vectorizer = load_model_and_vectorizer()

# Fungsi untuk membersihkan teks
def clean_text(text):
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = text.lower()  # Case folding
    words = word_tokenize(text)  # Tokenizing
    cleaned_words = [word for word in words if word not in stop_words]  # Stopword removal
    stemmed_words = [stemmer.stem(word) for word in cleaned_words]  # Stemming
    return " ".join(stemmed_words)

# Fungsi untuk melakukan klasifikasi teks
def classify_text(input_text):
    cleaned_text = clean_text(input_text)
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = logreg_model.predict(input_vector)[0]
    return predicted_label

# Fungsi untuk memasukkan data ke database
def insert_to_db(text, sentiment):
    conn = sqlite3.connect('db_scentplus.db')
    cursor = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''INSERT INTO riwayat (text, sentiment, date) VALUES (?, ?, ?)''', (text, sentiment, date))
    conn.commit()
    conn.close()

# Fungsi untuk mengambil data dari database
def fetch_data():
    conn = sqlite3.connect('db_scentplus.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT rowid AS id, text AS Text, sentiment, date FROM riwayat''')
    rows = cursor.fetchall()
    conn.close()
    return rows

# Fungsi untuk mengonversi DataFrame ke Excel
@st.cache(allow_output_mutation=True)
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
    processed_data = output.getvalue()
    return processed_data

# Fungsi untuk menjalankan aplikasi
def run():
    st.title("Aplikasi Analisis Sentimen Scentplus")

    tab1, tab2 = st.columns(2)

    with tab1:
        st.header("Masukkan kalimat untuk analisis sentimen:")
        input_text = st.text_input("Masukkan kalimat")
    
        if 'data' not in st.session_state:
            st.session_state['data'] = fetch_data()

        if st.button("Analisis"):
            if input_text.strip() == "":
                st.error("Tolong masukkan kalimat terlebih dahulu.")
            else:
                result = classify_text(input_text)
                st.write("Hasil Analisis Sentimen:", result)
                insert_to_db(input_text, result)
                st.session_state['data'] = fetch_data()
    
        # Menampilkan data dari database sebagai tabel dengan pagination
        data = st.session_state['data']
        if data:
            df = pd.DataFrame(data, columns=['id', 'Text', 'sentiment', 'date'])
            df.rename(columns={'sentiment': 'Human'}, inplace=True)
            
            # Konfigurasi AgGrid
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
            gridOptions = gb.build()

            try:
                AgGrid(
                    df,
                    gridOptions=gridOptions,
                    enable_enterprise_modules=True,
                    height=400,
                    fit_columns_on_grid_load=True
                )
            except Exception as e:
                st.error(f"Error menampilkan AgGrid: {e}")
            
            # Tombol untuk mengunduh data sebagai file Excel
            st.download_button(
                label="Unduh data sebagai file Excel",
                data=convert_df_to_excel(df),
                file_name="data_sentimen.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.write("Tidak ada data yang tersedia.")

    with tab2:
        st.header("Unggah file untuk Prediksi Sentimen")
        uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"], key="file_uploader")

        if uploaded_file is not None:
            # Baca file Excel
            df = pd.read_excel(uploaded_file)
            
            # Periksa apakah kolom 'Text' ada di file yang diunggah
            if 'Text' in df.columns:
                # Inisialisasi TF-IDF Vectorizer dan fit_transform pada data teks
                X = df['Text'].apply(clean_text)
                X_tfidf = tfidf_vectorizer.transform(X)
                
                # Lakukan prediksi
                df['Human'] = logreg_model.predict(X_tfidf)
                
                # Tampilkan prediksi
                st.write(df)
                
                # Buat tombol unduh
                st.download_button(
                    label="Unduh file dengan prediksi",
                    data=convert_df_to_excel(df),
                    file_name="prediksi_sentimen.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("File harus memiliki kolom 'Text'.")

if __name__ == "__main__":
    run()
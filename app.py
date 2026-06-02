from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# 1. Persiapan Resource NLP
nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('stopwords')

# 2. Pemuatan Model dan Alat Transformasi
try:
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model = joblib.load('models/best_model_svm.pkl') 
    label_encoder = joblib.load('models/label_encoder.pkl')
except FileNotFoundError:
    print("Peringatan: File model (.pkl) tidak ditemukan di folder 'models/'.")

# 3. Inisialisasi Sastrawi & Stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    """
    Fungsi ini harus identik dengan fungsi preprocessing saat tahap training
    untuk menjaga konsistensi data.
    """
    text = text.lower()
    # Hapus URL dan karakter non-alfabet
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    
    tokens = word_tokenize(text)
    # Stopword removal dan Stemming
    clean_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return " ".join(clean_tokens)

# 4. Rute Halaman Utama
@app.route('/')
def index():
    return render_template('index.html')

# Rute Halaman About Us
@app.route('/about')
def about():
    return render_template('about_us.html')

# 5. Rute Proses Prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        raw_text = request.form.get('news_text')
        
        if not raw_text:
            return render_template('index.html', error="Teks berita tidak boleh kosong.")
        # Tahap 1: Preprocessing
        processed_text = clean_text(raw_text)
        
        # Tahap 2: Vektorisasi TF-IDF
        vectorized_text = vectorizer.transform([processed_text]).toarray() 
        
        # Tahap 3: Prediksi menggunakan Model
        prediction_num = model.predict(vectorized_text)
        
        # Tahap 4: Mengembalikan label angka ke nama kategori (misal: 1 -> 'Ekonomi')
        category = label_encoder.inverse_transform(prediction_num)[0]
        
        return render_template('result.html', 
                               original_text=raw_text, 
                               prediction=category)

# Rute API
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    clean = clean_text(data['text'])
    vec = vectorizer.transform([clean]).toarray()
    pred = model.predict(vec)
    category = label_encoder.inverse_transform(pred)[0]
    

    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True)
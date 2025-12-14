from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import string
import os
import json
import csv
from datetime import datetime
from collections import defaultdict
import io
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Download NLTK data jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Kamus kata untuk setiap emosi (diperbanyak dan dikelompokkan)
happy_words = [
    'senang', 'bahagia', 'puas', 'gembira', 'optimis', 'positif', 'mantap', 
    'bagus', 'baik', 'lega', 'nyaman', 'tenang', 'bangga', 'sukses', 'hebat',
    'keren', 'memuaskan', 'menyenangkan', 'fantastis', 'luar biasa', 'cinta',
    'suka', 'antusias', 'semangat', 'percaya diri', 'puas hati', 'gembira ria'
]

sad_words = [
    'sedih', 'kecewa', 'frustasi', 'bingung', 'takut', 'khawatir', 'stress', 
    'tekanan', 'sulit', 'berat', 'rumit', 'pelik', 'masalah', 'kendala', 
    'hambatan', 'kesulitan', 'kesusahan', 'ketakutan', 'kekhawatiran', 'galau',
    'resah', 'gelisah', 'putus asa', 'pesimis', 'murung', 'frustrasi'
]

angry_words = [
    'marah', 'kesal', 'jengkel', 'benci', 'sebal', 'geram', 'frustasi', 
    'tidak adil', 'bosan', 'gemas', 'dongkol', 'berang', 'jengkel', 'kesal',
    'menjengkelkan', 'mengesalkan', 'memuakkan', 'menyebalkan', 'kesumat',
    'dendam', 'jengkel hati', 'marah besar'
]

# Database sederhana untuk menyimpan hasil
analysis_history = []
emotion_stats = defaultdict(lambda: {'count': 0, 'total_score': 0})

class EmotionAnalyzer:
    def __init__(self):
        self.happy_text = ' '.join(happy_words)
        self.sad_text = ' '.join(sad_words)
        self.angry_text = ' '.join(angry_words)
    
    def preprocess_text(self, text):
        """Preprocessing teks"""
        if not text:
            return ""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def calculate_vsm(self, comment):
        """Menghitung similarity menggunakan Vector Space Model"""
        try:
            processed_comment = self.preprocess_text(comment)
            
            if not processed_comment.strip():
                return {'happy': 33.33, 'sad': 33.33, 'angry': 33.33}
            
            # Gabungkan semua dokumen
            documents = [
                processed_comment,
                self.happy_text,
                self.sad_text,
                self.angry_text
            ]
            
            # Hitung TF-IDF
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Hitung similarity dengan setiap emosi
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            # Dapatkan skor similarity
            happy_score = float(similarities[0][0])
            sad_score = float(similarities[0][1])
            angry_score = float(similarities[0][2])
            
            # Normalisasi ke persentase (0-100%)
            scores = [happy_score, sad_score, angry_score]
            total = sum(scores)
            
            if total > 0:
                happy_percent = (happy_score / total) * 100
                sad_percent = (sad_score / total) * 100
                angry_percent = (angry_score / total) * 100
            else:
                # Jika semua similarity 0, beri distribusi merata
                happy_percent = sad_percent = angry_percent = 33.33
            
            return {
                'happy': round(happy_percent, 2),
                'sad': round(sad_percent, 2),
                'angry': round(angry_percent, 2)
            }
            
        except Exception as e:
            print(f"Error in VSM calculation: {e}")
            return {'happy': 33.33, 'sad': 33.33, 'angry': 33.33}
    
    def get_dominant_emotion(self, scores):
        """Mendapatkan emosi dominan"""
        max_score = max(scores.values())
        for emotion, score in scores.items():
            if score == max_score:
                return emotion.capitalize()
        return "Netral"
    
    def get_emotion_intensity(self, score):
        """Mendapatkan intensitas emosi"""
        if score >= 80:
            return "Sangat Tinggi"
        elif score >= 60:
            return "Tinggi"
        elif score >= 40:
            return "Sedang"
        elif score >= 20:
            return "Rendah"
        else:
            return "Sangat Rendah"

    # --- Text preprocessing helpers ---
    def tokenize_text(self, text):
        try:
            tokens = word_tokenize(text)
            return [t for t in tokens if t.strip()]
        except Exception:
            # Fallback: split on whitespace
            return [t for t in text.split() if t.strip()]

    def remove_stopwords(self, tokens):
        # Simple Indonesian stopword set (small but useful)
        stopwords = set([
            'saya','aku','kami','kamu','anda','dia','kita','mereka','yang','dan','di','ke','dari',
            'untuk','pada','ini','itu','adalah','ada','tidak','nya','atau','dengan','sebuah','sebagai',
            'itu','ini','dalam','oleh','ke','karena','agar','sehingga','juga','sudah','belum','masih'
        ])
        return [t for t in tokens if t.lower() not in stopwords]

    def stem_tokens(self, tokens):
        # Very simple rule-based stemmer removing common Indonesian suffixes
        suffixes = ['lah', 'kah', 'nya', 'ku', 'mu', 'kan', 'i', 'an']
        stems = []
        for t in tokens:
            word = t
            lowered = word.lower()
            changed = True
            # iteratively strip suffixes (limited times)
            for _ in range(2):
                for suf in suffixes:
                    if lowered.endswith(suf) and len(lowered) - len(suf) >= 3:
                        lowered = lowered[: -len(suf)]
                        break
                else:
                    changed = False
                    break
            stems.append(lowered)
        return stems

    def get_preprocessing_steps(self, text):
        cleaned = self.preprocess_text(text)
        tokens = self.tokenize_text(cleaned)
        tokens_no_sw = self.remove_stopwords(tokens)
        stems = self.stem_tokens(tokens_no_sw)
        return {
            'cleaning': cleaned,
            'tokenizing': tokens,
            'no_stopwords': tokens_no_sw,
            'stemming': stems
        }

# Initialize analyzer
analyzer = EmotionAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_emotion():
    try:
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'ok'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            return response
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400
            
        name = data.get('name', 'Anonymous')
        comment = data.get('comment', '')
        
        if not comment or not comment.strip():
            return jsonify({'error': 'Komentar tidak boleh kosong'}), 400
        
        # Calculate emotion scores
        scores = analyzer.calculate_vsm(comment)
        dominant_emotion = analyzer.get_dominant_emotion(scores)

        # Tentukan intensitas dengan aman (hindari KeyError jika label berbeda/bahasa)
        dominant_key = dominant_emotion.lower()
        if dominant_key not in scores:
            # Map beberapa kemungkinan label Indonesia ke kunci internal
            label_map = {'senang': 'happy', 'sedih': 'sad', 'marah': 'angry', 'netral': None}
            dominant_key = label_map.get(dominant_key, None)

        if dominant_key and dominant_key in scores:
            intensity = analyzer.get_emotion_intensity(scores[dominant_key])
        else:
            # Fallback: gunakan skor maksimum dari vektor sebagai dasar intensitas
            intensity = analyzer.get_emotion_intensity(max(scores.values()))

        # Preprocessing steps (cleaning, tokenizing, stopword removal, stemming)
        preprocessing = analyzer.get_preprocessing_steps(comment)
        
        # Simpan ke history
        analysis_data = {
            'id': len(analysis_history) + 1,
            'name': name,
            'comment': comment,
            'scores': scores,
            'dominant_emotion': dominant_emotion,
            'intensity': intensity,
            'preprocessing': preprocessing,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        analysis_history.append(analysis_data)
        
        # Update statistics
        for emotion, score in scores.items():
            emotion_stats[emotion]['count'] += 1
            emotion_stats[emotion]['total_score'] += score
        
        result = {
            'name': name,
            'comment': comment,
            'scores': scores,
            'dominant_emotion': dominant_emotion,
            'intensity': intensity,
            'preprocessing': preprocessing,
            'analysis_id': analysis_data['id'],
            'timestamp': analysis_data['timestamp'],
            'status': 'success'
        }

        # If the caller provided a true label, compute classification metrics for this sample
        try:
            true_label_raw = data.get('true_label') or data.get('label')
            if true_label_raw:
                def normalize_label(lbl):
                    if not lbl: return None
                    s = str(lbl).strip().lower()
                    map_en = {
                        'senang': 'happy', 'sedih': 'sad', 'marah': 'angry',
                        'happy': 'happy', 'sad': 'sad', 'angry': 'angry',
                        'positive': 'happy', 'pos': 'happy',
                        'negative': 'sad', 'neg': 'sad',
                        'neutral': None, 'netral': None
                    }
                    if s in map_en:
                        return map_en[s]
                    for k,v in map_en.items():
                        if k in s:
                            return v
                    return None

                y_true = []
                y_pred = []
                tkey = normalize_label(true_label_raw)
                pkey = None
                # predicted key
                det = dominant_emotion.lower() if dominant_emotion else ''
                if det in ['happy','sad','angry']:
                    pkey = det
                else:
                    label_map = {'senang': 'happy', 'sedih': 'sad', 'marah': 'angry'}
                    pkey = label_map.get(det, None)

                if tkey and pkey:
                    y_true.append(tkey)
                    y_pred.append(pkey)
                    # compute report using sklearn
                    rep = classification_report(y_true, y_pred, labels=['happy','sad','angry'], output_dict=True, zero_division=0)
                    result['metrics'] = rep
        except Exception as e:
            print('Error computing metrics for single sample:', e)
        
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"Error in analyze_emotion: {e}")
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/history')
def get_history():
    """Mendapatkan riwayat analisis"""
    return jsonify({
        'total_analyses': len(analysis_history),
        'history': analysis_history  # Kembalikan semua data, bukan hanya 10
    })

@app.route('/history/<int:analysis_id>')
def get_analysis_by_id(analysis_id):
    """Mendapatkan analisis spesifik berdasarkan ID"""
    try:
        # Cari analisis berdasarkan ID
        analysis = next((a for a in analysis_history if a['id'] == analysis_id), None)
        
        if not analysis:
            return jsonify({'error': 'Analisis tidak ditemukan'}), 404
            
        return jsonify(analysis)
        
    except Exception as e:
        print(f"Error in get_analysis_by_id: {e}")
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/statistics')
def statistics_page():
    """Halaman statistik dengan tampilan menarik"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Statistik - EmoAnalyzer</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            
            .header {
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                margin-bottom: 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 15px;
            }
            
            .header-title h1 {
                color: #333;
                font-size: 28px;
                margin-bottom: 5px;
            }
            
            .header-title p {
                color: #666;
                font-size: 14px;
            }
            
            .header-btn {
                padding: 10px 20px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 8px;
                text-decoration: none;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .header-btn:hover {
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .stat-card {
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .stat-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            }
            
            .stat-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea, #764ba2);
            }
            
            .stat-card.happy::before {
                background: linear-gradient(90deg, #fbbf24, #f59e0b);
            }
            
            .stat-card.sad::before {
                background: linear-gradient(90deg, #3b82f6, #1d4ed8);
            }
            
            .stat-card.angry::before {
                background: linear-gradient(90deg, #f87171, #dc2626);
            }
            
            .stat-card-icon {
                font-size: 40px;
                margin-bottom: 15px;
            }
            
            .stat-card.happy .stat-card-icon { color: #fbbf24; }
            .stat-card.sad .stat-card-icon { color: #3b82f6; }
            .stat-card.angry .stat-card-icon { color: #f87171; }
            .stat-card.total .stat-card-icon { color: #667eea; }
            
            .stat-value {
                font-size: 32px;
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }
            
            .stat-label {
                color: #666;
                font-size: 14px;
                margin-bottom: 10px;
            }
            
            .stat-detail {
                font-size: 12px;
                color: #999;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .charts-section {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                margin-bottom: 30px;
            }
            
            .charts-section h2 {
                color: #333;
                margin-bottom: 25px;
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 22px;
            }
            
            .charts-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 25px;
            }
            
            .chart-container {
                position: relative;
                height: 300px;
                padding: 15px;
                background: #f9fafb;
                border-radius: 10px;
            }
            
            .chart-title {
                font-weight: 600;
                color: #333;
                margin-bottom: 15px;
                font-size: 16px;
            }
            
            .distribution-section {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }
            
            .distribution-section h2 {
                color: #333;
                margin-bottom: 25px;
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 22px;
            }
            
            .distribution-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
            }
            
            .distribution-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                position: relative;
            }
            
            .distribution-card.happy {
                background: linear-gradient(135deg, #fff8e6 0%, #ffe4b3 100%);
                border-left: 4px solid #fbbf24;
            }
            
            .distribution-card.sad {
                background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
                border-left: 4px solid #3b82f6;
            }
            
            .distribution-card.angry {
                background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
                border-left: 4px solid #f87171;
            }
            
            .distribution-card h3 {
                color: #333;
                margin-bottom: 10px;
                font-size: 18px;
            }
            
            .distribution-value {
                font-size: 28px;
                font-weight: bold;
                color: #333;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
            }
            
            .distribution-value i {
                font-size: 32px;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #999;
            }
            
            .loading i {
                font-size: 32px;
                margin-bottom: 15px;
                display: block;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            
            @media (max-width: 768px) {
                .header {
                    flex-direction: column;
                    text-align: center;
                }
                
                .charts-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="header-title">
                    <h1><i class="fas fa-chart-bar"></i> Statistik Emosi</h1>
                    <p>Analisis menyeluruh tentang emosi mahasiswa</p>
                </div>
                <a href="/" class="header-btn">
                    <i class="fas fa-arrow-left"></i> Kembali
                </a>
            </div>
            
            <div class="stats-grid" id="statsGrid"></div>
            
            <div class="charts-section">
                <h2>
                    <i class="fas fa-chart-line"></i> Grafik Analisis
                </h2>
                <div class="charts-grid">
                    <div>
                        <div class="chart-title">Distribusi Emosi (Rata-rata %)</div>
                        <div class="chart-container">
                            <canvas id="emotionChart"></canvas>
                        </div>
                    </div>
                    <div>
                        <div class="chart-title">Emosi Dominan</div>
                        <div class="chart-container">
                            <canvas id="dominantChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="distribution-section">
                <h2>
                    <i class="fas fa-pie-chart"></i> Distribusi Emosi Dominan
                </h2>
                <div class="distribution-grid" id="distributionGrid"></div>
            </div>
        </div>
        
        <script>
            let emotionChart, dominantChart;
            
            async function loadStatistics() {
                try {
                    const response = await fetch('/statistics/json');
                    const data = await response.json();
                    
                    displayStats(data);
                    displayDistribution(data);
                    createCharts(data);
                } catch (error) {
                    console.error('Error loading statistics:', error);
                }
            }
            
            function displayStats(data) {
                const statsGrid = document.getElementById('statsGrid');
                const emotionStats = data.emotion_stats || {};
                const total = data.total_analyses || 0;
                
                let html = `
                    <div class="stat-card total">
                        <div class="stat-card-icon"><i class="fas fa-chart-line"></i></div>
                        <div class="stat-value">${total}</div>
                        <div class="stat-label">Total Analisis</div>
                        <div class="stat-detail">
                            <i class="fas fa-check-circle"></i> Data terekam
                        </div>
                    </div>
                `;
                
                if (emotionStats.happy) {
                    html += `
                        <div class="stat-card happy">
                            <div class="stat-card-icon">😊</div>
                            <div class="stat-value">${emotionStats.happy.average_score.toFixed(1)}%</div>
                            <div class="stat-label">Rata-rata Senang</div>
                            <div class="stat-detail">
                                <i class="fas fa-chart-line"></i> ${emotionStats.happy.total_occurrences} kemunculan
                            </div>
                        </div>
                    `;
                }
                
                if (emotionStats.sad) {
                    html += `
                        <div class="stat-card sad">
                            <div class="stat-card-icon">😔</div>
                            <div class="stat-value">${emotionStats.sad.average_score.toFixed(1)}%</div>
                            <div class="stat-label">Rata-rata Sedih</div>
                            <div class="stat-detail">
                                <i class="fas fa-chart-line"></i> ${emotionStats.sad.total_occurrences} kemunculan
                            </div>
                        </div>
                    `;
                }
                
                if (emotionStats.angry) {
                    html += `
                        <div class="stat-card angry">
                            <div class="stat-card-icon">😠</div>
                            <div class="stat-value">${emotionStats.angry.average_score.toFixed(1)}%</div>
                            <div class="stat-label">Rata-rata Marah</div>
                            <div class="stat-detail">
                                <i class="fas fa-chart-line"></i> ${emotionStats.angry.total_occurrences} kemunculan
                            </div>
                        </div>
                    `;
                }
                
                statsGrid.innerHTML = html;
            }
            
            function displayDistribution(data) {
                const distributionGrid = document.getElementById('distributionGrid');
                const dominantDist = data.dominant_distribution || {};
                
                let html = '';
                
                const emotionEmojis = {
                    'Happy': '😊',
                    'Sad': '😔',
                    'Angry': '😠'
                };
                
                const emotionClasses = {
                    'Happy': 'happy',
                    'Sad': 'sad',
                    'Angry': 'angry'
                };
                
                for (const [emotion, count] of Object.entries(dominantDist)) {
                    const emoji = emotionEmojis[emotion] || '😐';
                    const className = emotionClasses[emotion] || '';
                    html += `
                        <div class="distribution-card ${className}">
                            <h3>${emotion}</h3>
                            <div class="distribution-value">
                                <span>${count}</span>
                                <i>${emoji}</i>
                            </div>
                        </div>
                    `;
                }
                
                if (Object.keys(dominantDist).length === 0) {
                    html = '<p style="grid-column: 1/-1; text-align: center; color: #999;">Belum ada data</p>';
                }
                
                distributionGrid.innerHTML = html;
            }
            
            function createCharts(data) {
                const emotionStats = data.emotion_stats || {};
                
                // Emotion Average Chart
                const emotionLabels = [];
                const emotionValues = [];
                
                if (emotionStats.happy) {
                    emotionLabels.push('Senang');
                    emotionValues.push(emotionStats.happy.average_score);
                }
                if (emotionStats.sad) {
                    emotionLabels.push('Sedih');
                    emotionValues.push(emotionStats.sad.average_score);
                }
                if (emotionStats.angry) {
                    emotionLabels.push('Marah');
                    emotionValues.push(emotionStats.angry.average_score);
                }
                
                const emotionCtx = document.getElementById('emotionChart').getContext('2d');
                if (emotionChart) emotionChart.destroy();
                emotionChart = new Chart(emotionCtx, {
                    type: 'doughnut',
                    data: {
                        labels: emotionLabels,
                        datasets: [{
                            data: emotionValues,
                            backgroundColor: ['#fbbf24', '#3b82f6', '#f87171'],
                            borderColor: 'white',
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom'
                            }
                        }
                    }
                });
                
                // Dominant Distribution Chart
                const dominantDist = data.dominant_distribution || {};
                const dominantLabels = Object.keys(dominantDist);
                const dominantValues = Object.values(dominantDist);
                
                const dominantCtx = document.getElementById('dominantChart').getContext('2d');
                if (dominantChart) dominantChart.destroy();
                dominantChart = new Chart(dominantCtx, {
                    type: 'bar',
                    data: {
                        labels: dominantLabels,
                        datasets: [{
                            label: 'Jumlah Kemunculan',
                            data: dominantValues,
                            backgroundColor: ['#fbbf24', '#3b82f6', '#f87171'],
                            borderRadius: 8,
                            borderSkipped: false
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: false
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }
            
            window.addEventListener('load', loadStatistics);
            setInterval(loadStatistics, 5000);
        </script>
    </body>
    </html>
    """

@app.route('/statistics/json')
def get_statistics():
    """Mendapatkan statistik emosi (JSON API)"""
    stats = {}
    for emotion, data in emotion_stats.items():
        if data['count'] > 0:
            stats[emotion] = {
                'average_score': round(data['total_score'] / data['count'], 2),
                'total_occurrences': data['count']
            }
    
    # Hitung distribusi emosi dominan
    dominant_counts = defaultdict(int)
    for analysis in analysis_history:
        dominant_counts[analysis['dominant_emotion']] += 1
    
    return jsonify({
        'emotion_stats': stats,
        'dominant_distribution': dict(dominant_counts),
        'total_analyses': len(analysis_history)
    })

@app.route('/export/csv')
def export_csv():
    """Export data ke CSV"""
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['ID', 'Nama', 'Komentar', 'Senang (%)', 'Sedih (%)', 'Marah (%)', 'Emosi Dominan', 'Intensitas', 'Timestamp'])
        
        # Data
        for analysis in analysis_history:
            writer.writerow([
                analysis['id'],
                analysis['name'],
                analysis['comment'],
                analysis['scores']['happy'],
                analysis['scores']['sad'],
                analysis['scores']['angry'],
                analysis['dominant_emotion'],
                analysis['intensity'],
                analysis['timestamp']
            ])
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'analisis_emosi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export/json')
def export_json():
    """Export data ke JSON"""
    try:
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_analyses': len(analysis_history),
            'data': analysis_history
        }
        
        return send_file(
            io.BytesIO(json.dumps(export_data, indent=2, ensure_ascii=False).encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=f'analisis_emosi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """Halaman tentang"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tentang - Kelompok 3 RPL A 2022</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
            .back-btn { display: inline-block; margin-bottom: 20px; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-btn">← Kembali ke Beranda</a>
            <h1>📖 Tentang EmoAnalyzer</h1>
            <p><strong>EmoAnalyzer</strong> adalah sistem analisis emosi mahasiswa terhadap tugas akhir menggunakan metode <strong>Vector Space Model (VSM)</strong> berbasis <strong>Text Mining</strong>.</p>
            
            <h2>🎯 Tujuan</h2>
            <p>Membantu mengidentifikasi dan menganalisis emosi mahasiswa dalam menghadapi tugas akhir melalui analisis teks komentar mereka.</p>
            
            <h2>🔧 Teknologi</h2>
            <ul>
                <li><strong>Backend:</strong> Python Flask</li>
                <li><strong>Text Processing:</strong> NLTK, Scikit-learn</li>
                <li><strong>Algorithm:</strong> TF-IDF + Cosine Similarity (VSM)</li>
                <li><strong>Frontend:</strong> HTML, CSS, dan JavaScript</li>
            </ul>
            
            <h2>📊 Metode</h2>
            <p>Sistem menggunakan Vector Space Model untuk menghitung kemiripan antara komentar mahasiswa dengan vektor-vektor emosi (senang, sedih, marah) menggunakan Cosine Similarity.</p>
            
            <h2>👨‍💻 Pengembang</h2>
            <p>Sistem ini dikembangkan untuk tugas akhir mata kuliah Sistem Kembali Informasi.</p> 
            <p>Pengembang: <strong>Kelompok 3 RPL A 2022</strong></p>
            <ul>
                <li>Rendi Eko Kurniawan (NIM 22050974007)</li>
                <li>Lia Dwi Rusanti (NIM 22050974010)</li>
                <li>Jihan Salsabilah (NIM 22050974013)</li>
                <li>Siti Aulia Rahmadhani (NIM 22050974038)</li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.route('/tutorial')
def tutorial():
    """Halaman tutorial"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tutorial - EmoAnalyzer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
            .step { margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            .back-btn { display: inline-block; margin-bottom: 20px; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-btn">← Kembali ke Beranda</a>
            <h1>🎓 Tutorial Penggunaan</h1>
            
            <div class="step">
                <h3>1️⃣ Masukkan Data</h3>
                <p>Isi nama (opsional) dan komentar tentang perasaan Anda terhadap ketidakpastian informasi tugas akhir.</p>
            </div>
            
            <div class="step">
                <h3>2️⃣ Klik Analisis Emosi</h3>
                <p>Tekan tombol "Analisis Emosi" untuk memproses komentar Anda.</p>
            </div>
            
            <div class="step">
                <h3>3️⃣ Lihat Hasil</h3>
                <p>Sistem akan menampilkan tiga vektor emosi:
                <ul>
                    <li>😊 <strong>Vektor Senang</strong> - Tingkat kebahagiaan/positivitas</li>
                    <li>😔 <strong>Vektor Sedih</strong> - Tingkat kesedihan/kekhawatiran</li>
                    <li>😠 <strong>Vektor Marah</strong> - Tingkat kemarahan/frustasi</li>
                </ul>
                </p>
            </div>
            
            <div class="step">
                <h3>4️⃣ Ekspor Data</h3>
                <p>Anda dapat mengekspor hasil analisis dalam format CSV atau JSON untuk keperluan penelitian.</p>
            </div>
            
            <h2>💡 Tips</h2>
            <ul>
                <li>Gunakan bahasa Indonesia yang natural</li>
                <li>Jelaskan perasaan Anda secara detail</li>
                <li>Gunakan tombol contoh untuk testing cepat</li>
                <li>Cek statistik untuk melihat tren emosi</li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.route('/riwayat')
def riwayat():
    """Halaman riwayat analisis dengan tampilan menarik"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Riwayat Analisis - EmoAnalyzer</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1000px;
                margin: 0 auto;
            }
            
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                flex-wrap: wrap;
                gap: 15px;
            }
            
            .header-title {
                flex: 1;
                min-width: 200px;
            }
            
            .header-title h1 {
                color: #333;
                font-size: 28px;
                margin-bottom: 5px;
            }
            
            .header-title p {
                color: #666;
                font-size: 14px;
            }
            
            .header-buttons {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
            
            .btn {
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            
            .btn-back {
                background: #667eea;
                color: white;
            }
            
            .btn-back:hover {
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            .btn-export {
                background: #10b981;
                color: white;
            }
            
            .btn-export:hover {
                background: #059669;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(16, 185, 129, 0.4);
            }
            
            .stats-section {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }
            
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            }
            
            .stat-card i {
                font-size: 32px;
                margin-bottom: 10px;
                display: block;
            }
            
            .stat-card.happy i { color: #fbbf24; }
            .stat-card.sad i { color: #3b82f6; }
            .stat-card.angry i { color: #f87171; }
            .stat-card.total i { color: #667eea; }
            
            .stat-card h3 {
                font-size: 24px;
                color: #333;
                margin-bottom: 5px;
            }
            
            .stat-card p {
                color: #666;
                font-size: 13px;
            }
            
            .history-section {
                background: white;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                overflow: hidden;
            }
            
            .history-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px 25px;
                font-size: 20px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .history-empty {
                padding: 60px 20px;
                text-align: center;
                color: #999;
            }
            
            .history-empty i {
                font-size: 48px;
                color: #ddd;
                margin-bottom: 15px;
                display: block;
            }
            
            .history-list {
                display: flex;
                flex-direction: column;
                gap: 0;
            }
            
            .analysis-item {
                padding: 20px 25px;
                border-bottom: 1px solid #e5e7eb;
                transition: all 0.3s ease;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 15px;
            }
            
            .analysis-item:last-child {
                border-bottom: none;
            }
            
            .analysis-item:hover {
                background: #f9fafb;
            }
            
            .analysis-item-left {
                flex: 1;
            }
            
            .analysis-name {
                font-weight: 600;
                color: #333;
                margin-bottom: 5px;
            }
            
            .analysis-comment {
                color: #666;
                font-size: 14px;
                margin-bottom: 8px;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }
            
            .analysis-time {
                color: #999;
                font-size: 12px;
            }
            
            .analysis-item-right {
                display: flex;
                align-items: center;
                gap: 15px;
                flex-wrap: wrap;
                justify-content: flex-end;
            }
            
            .emotion-badge {
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 5px;
                white-space: nowrap;
            }
            
            .emotion-badge.happy {
                background: #fef3c7;
                color: #d97706;
            }
            
            .emotion-badge.sad {
                background: #dbeafe;
                color: #1e40af;
            }
            
            .emotion-badge.angry {
                background: #fee2e2;
                color: #dc2626;
            }
            
            .emotion-badge.netral {
                background: #e5e7eb;
                color: #374151;
            }
            
            .scores-display {
                display: flex;
                gap: 8px;
                font-size: 12px;
            }
            
            .score-item {
                background: #f3f4f6;
                padding: 4px 8px;
                border-radius: 4px;
                color: #666;
            }
            
            .score-item strong {
                color: #333;
                margin-right: 3px;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #999;
            }
            
            .loading i {
                font-size: 32px;
                margin-bottom: 15px;
                display: block;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            
            /* Responsive */
            @media (max-width: 768px) {
                .header {
                    flex-direction: column;
                    text-align: center;
                }
                
                .header-buttons {
                    justify-content: center;
                    width: 100%;
                }
                
                .analysis-item {
                    flex-direction: column;
                    align-items: flex-start;
                }
                
                .analysis-item-right {
                    width: 100%;
                    justify-content: flex-start;
                }
                
                .scores-display {
                    flex-wrap: wrap;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="header-title">
                    <h1><i class="fas fa-history"></i> Riwayat Analisis</h1>
                    <p>Lihat semua analisis emosi yang telah dilakukan</p>
                </div>
                <div class="header-buttons">
                    <a href="/" class="btn btn-back">
                        <i class="fas fa-arrow-left"></i> Kembali
                    </a>
                    <a href="/export/csv" class="btn btn-export">
                        <i class="fas fa-download"></i> Export CSV
                    </a>
                </div>
            </div>
            
            <div class="stats-section" id="statsSection"></div>
            
            <div class="history-section">
                <div class="history-header">
                    <i class="fas fa-chart-bar"></i>
                    Data Analisis
                </div>
                <div class="history-list" id="historyList">
                    <div class="loading">
                        <i class="fas fa-spinner"></i>
                        <p>Memuat data...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function loadData() {
                try {
                    const [historyResp, statsResp] = await Promise.all([
                        fetch('/history'),
                        fetch('/statistics/json')
                    ]);
                    
                    const historyData = await historyResp.json();
                    const statsData = await statsResp.json();
                    
                    displayStats(statsData);
                    displayHistory(historyData.history || []);
                } catch (error) {
                    console.error('Error loading data:', error);
                    document.getElementById('historyList').innerHTML = '<div class="history-empty"><i class="fas fa-exclamation-circle"></i><p>Gagal memuat data</p></div>';
                }
            }
            
            function displayStats(stats) {
                const statsSection = document.getElementById('statsSection');
                const emotionStats = stats.emotion_stats || {};
                const totalAnalyses = stats.total_analyses || 0;
                
                let html = `
                    <div class="stat-card total">
                        <i class="fas fa-chart-line"></i>
                        <h3>${totalAnalyses}</h3>
                        <p>Total Analisis</p>
                    </div>
                `;
                
                if (emotionStats.happy) {
                    html += `
                        <div class="stat-card happy">
                            <i class="fas fa-smile"></i>
                            <h3>${emotionStats.happy.average_score.toFixed(1)}%</h3>
                            <p>Rata-rata Senang</p>
                        </div>
                    `;
                }
                
                if (emotionStats.sad) {
                    html += `
                        <div class="stat-card sad">
                            <i class="fas fa-frown"></i>
                            <h3>${emotionStats.sad.average_score.toFixed(1)}%</h3>
                            <p>Rata-rata Sedih</p>
                        </div>
                    `;
                }
                
                if (emotionStats.angry) {
                    html += `
                        <div class="stat-card angry">
                            <i class="fas fa-angry"></i>
                            <h3>${emotionStats.angry.average_score.toFixed(1)}%</h3>
                            <p>Rata-rata Marah</p>
                        </div>
                    `;
                }
                
                statsSection.innerHTML = html;
            }
            
            function displayHistory(history) {
                const historyList = document.getElementById('historyList');
                
                if (!history || history.length === 0) {
                    historyList.innerHTML = '<div class="history-empty"><i class="fas fa-inbox"></i><p>Tidak ada data analisis</p></div>';
                    return;
                }
                
                let html = '';
                history.forEach((item, index) => {
                    const emotionClass = item.dominant_emotion.toLowerCase();
                    const emotionIcon = getEmotionIcon(emotionClass);
                    
                    html += `
                        <div class="analysis-item">
                            <div class="analysis-item-left">
                                <div class="analysis-name">
                                    <i class="fas fa-user"></i> ${escapeHtml(item.name || 'Anonymous')}
                                </div>
                                <div class="analysis-comment">"${escapeHtml(item.comment)}"</div>
                                <div class="analysis-time">
                                    <i class="fas fa-clock"></i> ${item.timestamp}
                                </div>
                            </div>
                            <div class="analysis-item-right">
                                <div class="emotion-badge ${emotionClass}">
                                    ${emotionIcon} ${item.dominant_emotion}
                                </div>
                                <div class="scores-display">
                                    <div class="score-item">
                                        <strong>😊</strong> ${(item.scores.happy * 100).toFixed(0)}%
                                    </div>
                                    <div class="score-item">
                                        <strong>😔</strong> ${(item.scores.sad * 100).toFixed(0)}%
                                    </div>
                                    <div class="score-item">
                                        <strong>😠</strong> ${(item.scores.angry * 100).toFixed(0)}%
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                historyList.innerHTML = html;
            }
            
            function getEmotionIcon(emotion) {
                const icons = {
                    'senang': '😊',
                    'sedih': '😔',
                    'marah': '😠',
                    'netral': '😐'
                };
                return icons[emotion] || '😐';
            }
            
            function escapeHtml(str) {
                if (!str) return '';
                return String(str)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#39;');
            }
            
            // Load data on page load
            window.addEventListener('load', loadData);
            
            // Reload data every 5 seconds to show updates
            setInterval(loadData, 5000);
        </script>
    </body>
    </html>
    """

@app.route('/test')
def test():
    return "✅ Server Flask berjalan dengan baik!"

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Server is running!'})

@app.route('/analyze/csv', methods=['POST', 'OPTIONS'])
def analyze_csv():
    try:
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'ok'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            return response
            
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
        
        # Check if file is CSV
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format file tidak didukung. Harus CSV'}), 400
        
        # Read CSV file
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Error membaca file CSV: {str(e)}'}), 400
        
        # Check required columns
        if 'comment' not in df.columns:
            return jsonify({'error': 'File CSV harus memiliki kolom "comment"'}), 400
        
        # Process each comment
        results = []
        total_rows = len(df)
        analysis_ids = []  # Simpan ID analisis yang baru dibuat
        y_true = []
        y_pred = []
        
        for index, row in df.iterrows():
            comment = str(row['comment']).strip()
            name = str(row.get('name', 'Anonymous')).strip() if 'name' in df.columns else 'Anonymous'
            
            if not comment:
                continue
            
            # Calculate emotion scores
            scores = analyzer.calculate_vsm(comment)
            dominant_emotion = analyzer.get_dominant_emotion(scores)

            # Tentukan intensitas dengan aman (hindari KeyError jika label berbeda/bahasa)
            dominant_key = dominant_emotion.lower()
            if dominant_key not in scores:
                label_map = {'senang': 'happy', 'sedih': 'sad', 'marah': 'angry', 'netral': None}
                dominant_key = label_map.get(dominant_key, None)

            if dominant_key and dominant_key in scores:
                intensity = analyzer.get_emotion_intensity(scores[dominant_key])
            else:
                intensity = analyzer.get_emotion_intensity(max(scores.values()))

            # Preprocessing
            preprocessing = analyzer.get_preprocessing_steps(comment)
            
            # Save to history
            analysis_data = {
                'id': len(analysis_history) + 1,
                'name': name,
                'comment': comment,
                'scores': scores,
                'dominant_emotion': dominant_emotion,
                'intensity': intensity,
                'preprocessing': preprocessing,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'source': 'csv_upload'
            }
            analysis_history.append(analysis_data)
            
            # Update statistics
            for emotion, score in scores.items():
                emotion_stats[emotion]['count'] += 1
                emotion_stats[emotion]['total_score'] += score
            
            analysis_ids.append(analysis_data['id'])  # Simpan ID
            
            results.append({
                'row_number': index + 1,
                'name': name,
                'comment': comment,
                'scores': scores,
                'dominant_emotion': dominant_emotion,
                'intensity': intensity,
                'analysis_id': analysis_data['id']
            })
            # collect labels for metrics if provided in CSV
            label_col = None
            if 'label' in df.columns:
                label_col = 'label'
            elif 'true_label' in df.columns:
                label_col = 'true_label'

            if label_col:
                raw_true = row.get(label_col, None)
                def normalize_label(lbl):
                    if not lbl: return None
                    s = str(lbl).strip().lower()
                    map_en = {
                        'senang': 'happy', 'sedih': 'sad', 'marah': 'angry',
                        'happy': 'happy', 'sad': 'sad', 'angry': 'angry',
                        'positive': 'happy', 'pos': 'happy',
                        'negative': 'sad', 'neg': 'sad',
                        'neutral': None, 'netral': None
                    }
                    if s in map_en:
                        return map_en[s]
                    for k,v in map_en.items():
                        if k in s:
                            return v
                    return None

                tkey = normalize_label(raw_true)
                det = dominant_emotion.lower() if dominant_emotion else ''
                pkey = det if det in ['happy','sad','angry'] else None
                if tkey and pkey:
                    y_true.append(tkey)
                    y_pred.append(pkey)
        
        response_data = {
            'status': 'success',
            'message': f'Berhasil menganalisis {len(results)} dari {total_rows} komentar',
            'total_processed': len(results),
            'total_rows': total_rows,
            'results': results,
            'analysis_ids': analysis_ids  # Kembalikan juga daftar ID
        }
        # Jika ada label sebenarnya pada CSV, hitung metrik keseluruhan
        try:
            if len(y_true) > 0:
                rep = classification_report(y_true, y_pred, labels=['happy','sad','angry'], output_dict=True, zero_division=0)
                response_data['metrics'] = rep
        except Exception as e:
            print('Error computing metrics for CSV upload:', e)

        # Detect split column (train/test) in CSV and count samples. If absent, perform a default split (80/20).
        try:
            split_col = None
            for c in ['split', 'set', 'subset', 'type', 'partition', 'stage', 'split_label', 'is_train']:
                if c in df.columns:
                    split_col = c
                    break

            if split_col:
                train_kw = ['train', 'training', 'latih', 'trainset']
                test_kw = ['test', 'testing', 'uji', 'testset']
                train_count = 0
                test_count = 0
                unknown_count = 0
                for _, row in df.iterrows():
                    v = row.get(split_col, '')
                    s = str(v).strip().lower()
                    if any(k == s or k in s for k in train_kw):
                        train_count += 1
                    elif any(k == s or k in s for k in test_kw):
                        test_count += 1
                    else:
                        unknown_count += 1

                response_data['split_counts'] = {
                    'train': train_count,
                    'test': test_count,
                    'unknown': unknown_count,
                    'total_rows': int(total_rows)
                }
            else:
                # No split column: perform default random split (80% train, 20% test).
                try:
                    label_col = None
                    if 'label' in df.columns:
                        label_col = 'label'
                    elif 'true_label' in df.columns:
                        label_col = 'true_label'

                    stratify_vals = None
                    if label_col and df[label_col].nunique() > 1:
                        # use values as-is for stratification
                        stratify_vals = df[label_col]

                    # Use indices to split to avoid copying large data
                    indices = df.index.to_list()
                    if len(indices) <= 1:
                        train_count = len(indices)
                        test_count = 0
                    else:
                        try:
                            train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=stratify_vals if stratify_vals is not None else None, random_state=42)
                            train_count = len(train_idx)
                            test_count = len(test_idx)
                        except Exception:
                            # fallback to simple split
                            split_at = int(len(indices) * 0.8)
                            train_count = split_at
                            test_count = len(indices) - split_at

                    response_data['split_counts'] = {
                        'train': int(train_count),
                        'test': int(test_count),
                        'unknown': 0,
                        'total_rows': int(total_rows)
                    }
                except Exception as e:
                    print('Error computing default split:', e)
                    response_data['split_counts'] = {
                        'train': 0,
                        'test': 0,
                        'unknown': int(total_rows),
                        'total_rows': int(total_rows)
                    }
        except Exception as e:
            print('Error detecting split counts:', e)
        
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"Error in analyze_csv: {e}")
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

# Tambahkan route untuk template upload CSV
@app.route('/upload')
def upload_page():
    """Halaman upload CSV"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload CSV - EmoAnalyzer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
            .back-btn { display: inline-block; margin-bottom: 20px; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; }
            .upload-area { border: 2px dashed #667eea; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; }
            .btn { display: inline-flex; align-items: center; gap: 0.5rem; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 8px; cursor: pointer; transition: all 0.3s; text-decoration: none; font-weight: 500; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); }
            .loading { display: none; text-align: center; padding: 20px; }
            .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 1rem; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .results { display: none; margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-btn">← Kembali ke Beranda</a>
            <h1>📁 Analisis Emosi dari CSV</h1>
            
            <div class="upload-area">
                <h3>📊 Unggah File CSV</h3>
                <p>File CSV harus memiliki kolom <strong>"comment"</strong> (wajib) dan <strong>"name"</strong> (opsional)</p>
                
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="csvFile" name="file" accept=".csv" required style="margin: 20px 0;">
                    <br>
                    <button type="submit" class="btn">
                        <i class="fas fa-upload"></i> Upload & Analisis
                    </button>
                </form>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Menganalisis file CSV...</p>
            </div>

            <div class="results" id="results">
                <h3>📈 Hasil Analisis</h3>
                <div id="resultContent"></div>
            </div>

            <div style="margin-top: 30px; padding: 20px; background: #e7f3ff; border-radius: 10px;">
                <h4>📋 Format CSV yang Didukung:</h4>
                <pre style="background: white; padding: 15px; border-radius: 5px; overflow-x: auto;">
name,comment
John Doe,"Saya senang dengan bimbingan dosen"
Jane Smith,"Saya sedih karena jadwal tidak jelas"
...</pre>
                <p><strong>Note:</strong> Kolom "name" bersifat opsional. Jika tidak ada, akan digunakan "Anonymous"</p>
            </div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('csvFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Pilih file CSV terlebih dahulu!');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                try {
                    const response = await fetch('/analyze/csv', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(result.error || 'Terjadi kesalahan');
                    }
                    
                    // Show results
                    document.getElementById('resultContent').innerHTML = `
                        <div style="background: #d4edda; color: #155724; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                            <strong>✅ ${result.message}</strong>
                        </div>
                        <p><strong>Total Baris:</strong> ${result.total_rows}</p>
                        <p><strong>Berhasil Diproses:</strong> ${result.total_processed}</p>
                        <p><strong>Gagal:</strong> ${result.total_rows - result.total_processed}</p>
                        
                        ${result.analysis_ids && result.analysis_ids.length > 0 ? `
                        <div style="margin-top: 15px;">
                            <p><strong>Analisis Terbaru:</strong></p>
                            <div style="max-height: 150px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                                ${result.analysis_ids.map(id => `
                                    <div style="margin-bottom: 5px;">
                                        <a href="/?analysis_id=${id}" target="_blank" style="color: #667eea; text-decoration: none;">
                                            <i class="fas fa-external-link-alt"></i> Lihat Analisis ID: ${id}
                                        </a>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        ` : ''}
                        
                        <div style="margin-top: 20px;">
                            <a href="/export/csv" class="btn">
                                <i class="fas fa-download"></i> Export Hasil CSV
                            </a>
                            <a href="/" class="btn" style="background: #6c757d;">
                                <i class="fas fa-home"></i> Kembali ke Beranda
                            </a>
                        </div>
                    `;
                    
                    document.getElementById('results').style.display = 'block';
                    
                    // Jika backend mengembalikan metrics, tampilkan juga dan simpan untuk halaman utama
                    try {
                        if (result.metrics) {
                            const metricsHtml = `<div style="margin-top:15px;"><h4>Metrik Keseluruhan (CSV)</h4><pre style="background:#fff;padding:10px;border-radius:6px;overflow:auto;">${JSON.stringify(result.metrics, null, 2)}</pre></div>`;
                            document.getElementById('resultContent').insertAdjacentHTML('beforeend', metricsHtml);
                            localStorage.setItem('last_metrics', JSON.stringify(result.metrics));
                        }

                        if (result.split_counts) {
                            localStorage.setItem('last_split_counts', JSON.stringify(result.split_counts));
                        }

                        localStorage.setItem('csv_upload_completed', new Date().getTime().toString());
                    } catch (e) {
                        console.log('localStorage not available');
                    }
                    
                    // Redirect ke beranda setelah 2 detik
                    setTimeout(function() {
                        window.location.href = '/';
                    }, 2000);
                    
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """

@app.route('/history/clear', methods=['POST', 'OPTIONS'])
def clear_history():
    """Menghapus semua riwayat analisis"""
    try:
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'ok'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            return response
            
        # Hapus semua riwayat
        analysis_history.clear()
        
        # Reset statistik
        emotion_stats.clear()
        
        response_data = {
            'status': 'success',
            'message': 'Semua riwayat analisis berhasil dihapus',
            'total_analyses': 0
        }
        
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"Error in clear_history: {e}")
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("SISTEM ANALISIS EMOSI MAHASISWA - WEBSITE UTUH")
    print("=" * 60)
    print("URL Akses: http://localhost:5000")
    print("Statistics: http://localhost:5000/statistics")
    print("History: http://localhost:5000/history")
    print("About: http://localhost:5000/about")
    print("Tutorial: http://localhost:5000/tutorial")
    print("=" * 60)
    print("Tekan CTRL+C untuk menghentikan server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
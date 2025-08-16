import React, { useState } from 'react';
import './App.css';

function App() {
  const [newsText, setNewsText] = useState('');
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleCheck = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/process_text', {
        method: 'POST',
        headers: {
          'Content-Type': 'text/plain',
        },
        body: newsText.trim(),
      });

      const data = await response.json();
      setIsLoading(false);

      if (!response.ok) {
        setResult('حدث خطأ في الاتصال بالخادم.');
      } else {
        setResult(data.message);
      }
    } catch (err) {
      setIsLoading(false);
      setResult('حدث خطأ في الاتصال بالخادم أو فشل في الشبكة.');
    }
  };

  return (
    <div className="app">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;700&display=swap');

        body {
          margin: 0;
          font-family: 'IBM Plex Sans Arabic', sans-serif;
          background-color: #111827;
          color: #ffffff;
          direction: rtl;
        }

        .header {
          text-align: center;
          padding: 30px 20px;
          background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
          color: white;
          font-size: 2.8rem;
          font-weight: 700;
          letter-spacing: 1px;
          border-bottom: 4px solid #7c3aed;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }

        .main {
          max-width: 800px;
          margin: 60px auto;
          background-color: #1f2937;
          padding: 40px;
          border-radius: 20px;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }

        .news-input {
          width: 95%;
          min-height: 200px;
          padding: 20px;
          font-size: 1.1rem;
          background-color: #111827;
          border: 2px solid #8b5cf6;
          border-radius: 12px;
          color: #f3f4f6;
          resize: vertical;
          margin-bottom: 30px;
          outline: none;
          transition: 0.3s ease;
        }

        .news-input:focus {
          border-color: #0ea5e9;
          box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.3);
        }

        .center-button {
          display: flex;
          justify-content: center;
          margin-top: 20px;
        }

        .check-button {
          background: linear-gradient(to right, #8b5cf6, #0ea5e9);
          color: white;
          font-size: 1.2rem;
          font-weight: bold;
          padding: 14px 36px;
          border: none;
          border-radius: 50px;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4);
        }

        .check-button:hover {
          transform: scale(1.05);
        }

        .check-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .result-box {
          background: rgba(139, 92, 246, 0.1);
          border-right: 5px solid #8b5cf6;
          padding: 25px;
          margin-top: 40px;
          border-radius: 14px;
          box-shadow: 0 0 15px rgba(139, 92, 246, 0.3);
        }

        .result-box h3 {
          margin-top: 0;
          color: #a78bfa;
        }

        .footer {
          margin-top: 80px;
          text-align: center;
          padding: 25px;
          font-size: 0.95rem;
          color: #9ca3af;
          background-color: #1f2937;
          border-top: 1px solid #374151;
        }

        .overlay {
          position: fixed;
          top: 0; right: 0; bottom: 0; left: 0;
          background: rgba(17, 24, 39, 0.9);
          z-index: 9999;
          display: flex;
          justify-content: center;
          align-items: center;
          font-size: 1.8rem;
          color: #0ea5e9;
          font-weight: bold;
          backdrop-filter: blur(6px);
        }

        @media (max-width: 600px) {
          .header {
            font-size: 2rem;
          }

          .main {
            margin: 30px 15px;
            padding: 25px;
          }
        }
      `}</style>

      {isLoading && (
        <div className="overlay">جارٍ التحقق من الخبر...</div>
      )}

      <header className="header">عين الحقيقة - فلسطين</header>

      <main className="main">
        <textarea
          className="news-input"
          placeholder="أدخل الخبر الذي ترغب في تحليله والتحقق منه..."
          value={newsText}
          onChange={(e) => setNewsText(e.target.value)}
        />

        <div className="center-button">
          <button
            className="check-button"
            onClick={handleCheck}
            disabled={isLoading || !newsText.trim()}
          >
            {isLoading ? 'جارٍ التحقق...' : 'تحقق الآن'}
          </button>
        </div>

        {result && (
          <div className="result-box">
            <h3>النتيجة:</h3>
            <p>{result}</p>
          </div>
        )}
      </main>

      <footer className="footer">
        &copy; {new Date().getFullYear()} عين الحقيقة - فلسطين. جميع الحقوق محفوظة.
      </footer>
    </div>
  );
}

export default App;

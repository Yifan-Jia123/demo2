import { useState } from 'react';
import './App.css';

interface RecommendationResponse {
  predictions: Record<string, number>;
  recommendation: string;
}

const MOVIE_LIST = ["复仇者联盟", "蜘蛛侠", "泰坦尼克号", "爱情故事"];

function App() {
  // 状态：动态记录用户对各电影的打分，初始都为 null
  const [ratings, setRatings] = useState<Record<string, number | null>>({
    "复仇者联盟": null,
    "蜘蛛侠": null,
    "泰坦尼克号": null,
    "爱情故事": null
  });

  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // 处理下拉框打分变化
  const handleRatingChange = (movie: string, value: string) => {
    const numValue = value === "" ? null : Number(value);
    setRatings(prev => ({ ...prev, [movie]: numValue }));
  };

  const fetchRecommendation = async () => {
    setLoading(true);
    setResult(null);
    try {
      // 改为 POST 请求，把当前的打分当做 JSON 传给后端
      // 注意：如果你正在局域网展示，请把 127.0.0.1 换成你的 IP 地址
      const response = await fetch(`http://127.0.0.1:8000/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ ratings })
      });
      const data: RecommendationResponse = await response.json();
      setResult(data);
    } catch (error) {
      console.error("请求失败", error);
    } finally {
      setLoading(false);
    }
  };

  return (
      <div className="app-container">
        <div className="main-card">
          <h1 className="title" style={{ color: '#ffffff' }}>🎬 SVD 动态推荐引擎</h1>
          <p className="subtitle">录入你的口味，让隐语义算法懂你</p>

          {/* 新的控制面板：动态打分输入区 */}
          <div className="dynamic-input-section">
            <h3 className="list-title">✍️ 请对看过的电影打分：</h3>
            <div className="raw-data-grid">
              {MOVIE_LIST.map((movie) => (
                  <div className="raw-data-card" key={movie}>
                    <div className="raw-movie-name">{movie}</div>
                    <select
                        className="rating-select"
                        value={ratings[movie] === null ? "" : ratings[movie]}
                        onChange={(e) => handleRatingChange(movie, e.target.value)}
                    >
                      <option value="">未看</option>
                      <option value="1">1 星 ⭐️</option>
                      <option value="2">2 星 ⭐️⭐️</option>
                      <option value="3">3 星 ⭐️⭐️⭐️</option>
                      <option value="4">4 星 ⭐️⭐️⭐️⭐️</option>
                      <option value="5">5 星 ⭐️⭐️⭐️⭐️⭐️</option>
                    </select>
                  </div>
              ))}
            </div>

            <button
                onClick={fetchRecommendation}
                className="action-button"
                disabled={loading}
                style={{ width: '100%', marginTop: '20px' }}
            >
              {loading ? '正在进行 SVD 矩阵运算...' : '注入算法，生成专属推荐 ✨'}
            </button>
          </div>

          {/* 结果展示区 */}
          {result && (
              <div className="result-container fade-in">
                <div className="predictions-list">
                  <h3 className="list-title">📊 算法潜在偏好预测 (0-5分)：</h3>
                  {Object.entries(result.predictions).map(([movie, score]) => (
                      <div className="prediction-item" key={movie}>
                        <span className="movie-name">{movie}</span>
                        <div className="score-bar-container">
                          <div
                              className="score-bar"
                              style={{ width: `${(score / 5) * 100}%` }}
                          ></div>
                        </div>
                        <span className="score-value">{score.toFixed(2)}</span>
                      </div>
                  ))}
                </div>

                <div className="recommendation-highlight">
                  <h3>🏆 SVD 强推 (滤除已看)</h3>
                  <div className="top-movie">
                    🍿 《{result.recommendation}》
                  </div>
                </div>
              </div>
          )}
        </div>
      </div>
  );
}

export default App;
import { useState } from 'react';
import './App.css';

// 1. 定义数据类型
const MOVIE_NAMES = [
  "复仇者联盟", "蜘蛛侠", "星际穿越", "流浪地球",
  "泰坦尼克号", "初恋这件小事", "情书", "假如爱有天意",
  "盗梦空间", "黑客帝国"
];

interface RecommendationItem {
  movie: string;
  score: number;
  type: string;
  is_observed?: boolean;
}

interface RecommendationResponse {
  user: string;
  results: RecommendationItem[];
  explanation?: string;
  error?: string;
}

function App() {
  // --- 状态管理 ---
  const [username, setUsername] = useState<string>('张三');
  const [userRatings, setUserRatings] = useState<Record<string, number>>({});
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // --- 逻辑函数 1：获取已有用户的推荐 ---
  const fetchRecommendation = async () => {
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch(`http://127.0.0.1:8000/recommend/${username}`);
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("请求失败", error);
    } finally {
      setLoading(false);
    }
  };

  // --- 逻辑函数 2：提交新用户打分并实时计算 (核心交互) ---
  const submitNewUserRatings = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/recommend/new_user`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ratings: userRatings })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("提交失败，请检查后端是否开启");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="main-card">
        <h1 className="title">🎬 BiasSVD 矩阵分解引擎</h1>
        <p className="subtitle">基于 10x10 稀疏矩阵的特征提取与实时推荐</p>

        {/* 交互区 A：已有用户查询 */}
        <div className="control-panel">
          <select value={username} onChange={(e) => setUsername(e.target.value)} className="user-select">
            {["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十"].map(u => (
              <option key={u} value={u}>{u}</option>
            ))}
          </select>
          <button onClick={fetchRecommendation} className="action-button" disabled={loading}>
            已有用户查询
          </button>
        </div>

        <hr style={{ border: '0.5px solid #333', margin: '20px 0' }} />

        {/* 交互区 B：新用户实时交互 (你的新需求) */}
        <div className="input-section" style={{ background: '#1a1a1a', padding: '15px', borderRadius: '10px' }}>
          <h3 style={{ color: '#e67e22' }}>✨ 实时交互：输入您的打分</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            {MOVIE_NAMES.map(movie => (
              <div key={movie} style={{ display: 'flex', justifyContent: 'space-between', color: '#ccc' }}>
                <span style={{ fontSize: '13px' }}>{movie}:</span>
                <input
                  type="number" min="0" max="5" placeholder="0-5"
                  style={{ width: '50px', background: '#333', color: '#fff', border: 'none' }}
                  onChange={(e) => setUserRatings({ ...userRatings, [movie]: parseInt(e.target.value) || 0 })}
                />
              </div>
            ))}
          </div>
          <button onClick={submitNewUserRatings} className="action-button" style={{ marginTop: '15px', width: '100%', background: '#e67e22' }}>
            提交并实时训练矩阵
          </button>
        </div>

        {/* 可视化展示区 */}
        <div className="visualize-section" style={{ marginTop: '20px', textAlign: 'center' }}>
          <h3 style={{ color: '#fff' }}>🧬 隐空间特征分布图 (Latent Space)</h3>
          <img
            src={`http://127.0.0.1:8000/visualize?t=${new Date().getTime()}`}
            alt="SVD Visualization"
            style={{ width: '100%', borderRadius: '12px', border: '1px solid #444' }}
          />
        </div>

        {/* 结果展示区 */}
        {result && (
          <div className="result-container fade-in">
            <h3 className="list-title">📊 矩阵补全结果 ({result.user})：</h3>
            {result.results?.map((item) => (
              <div className="prediction-item" key={item.movie}>
                <div className="movie-info">
                  <span className="movie-name">{item.movie}</span>
                  <span className="movie-tag">[{item.type}]</span>
                </div>
                <div className="score-bar-container">
                  <div className="score-bar" style={{ width: `${item.score * 20}%` }}></div>
                </div>
                <span className="score-value">{item.score.toFixed(2)}分</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
import numpy as np
import datetime
import io
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel

class UserRating(BaseModel):
    ratings: dict # 格式如 {"复仇者联盟": 5, "情书": 1}




app = FastAPI(title="矩阵分解应用：基于 BiasSVD 的电影推荐")

# 在 app = FastAPI() 之后添加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，开发阶段最方便
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/recommend/new_user")
def predict_for_new_user(data: UserRating):
    # 1. 初始化一个临时的用户偏置和特征向量
    temp_bu = 0.0
    temp_pu = np.random.normal(scale=1./model.k, size=(model.k))

    # 2. 针对该用户的输入，进行少量迭代（快速收敛）
    for _ in range(50):
        for m_name, rating in data.ratings.items():
            if m_name in MOVIE_NAMES:
                i = MOVIE_NAMES.index(m_name)
                pred = model.mu + temp_bu + model.bi[i] + np.dot(temp_pu, model.Q[i])
                error = rating - pred
                # 仅更新该用户的参数
                temp_bu += model.lr * (error - model.lam * temp_bu)
                temp_pu += model.lr * (error * model.Q[i] - model.lam * temp_pu)

    # 3. 基于更新后的向量生成全量电影预测
    results = []
    for i, m_name in enumerate(MOVIE_NAMES):
        score = model.mu + temp_bu + model.bi[i] + np.dot(temp_pu, model.Q[i])
        results.append({"movie": m_name, "score": round(float(score), 2), "type": MOVIE_TAGS[m_name]})

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"user": "新用户", "results": results}


# ==========================================
# 1. 原始观测矩阵 R (含有缺失值 0)
# ==========================================
# 矩阵结构：行=用户，列=电影

# 后端更新：模拟更复杂的稀疏矩阵
USER_NAMES = ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十", "郑一", "冯二"]
MOVIE_NAMES = [
    "复仇者联盟", "蜘蛛侠", "星际穿越", "流浪地球",  # 动作/科幻组
    "泰坦尼克号", "初恋这件小事", "情书", "假如爱有天意",  # 爱情组
    "盗梦空间", "黑客帝国"  # 烧脑组
]

# 模拟 10x10 矩阵，设置 70% 的空缺 (0) 以体现稀疏性
# 这里的分布要有规律，比如前几个用户偏好动作，后几个偏好爱情
R = np.array([
    [5, 4, 5, 0, 0, 0, 0, 0, 5, 0],  # 用户0: 动作硬核
    [0, 5, 0, 4, 0, 0, 0, 0, 0, 5],
    [0, 0, 0, 0, 5, 5, 4, 0, 0, 0],  # 用户2: 爱情专家
    [0, 0, 0, 1, 4, 0, 5, 5, 0, 0],
    [5, 0, 4, 0, 0, 1, 0, 0, 5, 5],  # 用户4: 动作+烧脑
    [0, 0, 0, 0, 5, 4, 0, 5, 0, 0],
    [4, 5, 0, 5, 0, 0, 0, 0, 4, 0],
    [1, 0, 0, 0, 5, 5, 5, 4, 0, 0],
    [0, 4, 5, 5, 0, 0, 0, 0, 5, 5],
    [5, 5, 5, 0, 0, 1, 0, 0, 0, 5]
], dtype=float)

MOVIE_TAGS = {
    "复仇者联盟": "动作", "蜘蛛侠": "动作", "星际穿越": "科幻", "流浪地球": "科幻",
    "泰坦尼克号": "爱情", "初恋这件小事": "爱情", "情书": "爱情", "假如爱有天意": "爱情",
    "盗梦空间": "烧脑", "黑客帝国": "烧脑"
}


# ==========================================
# 2. 核心算法：BiasSVD 实现
# ==========================================
class BiasSVDRecommender:
    def __init__(self, R, k=2, lr=0.02, lam=0.02):
        self.R = R
        self.m, self.n = R.shape
        self.k = k      # 隐因子维度（降维后的特征数）
        self.lr = lr    # 学习率
        self.lam = lam  # 正则化系数（防止矩阵元素过大导致的过拟合）

        # A. 全局偏置：所有已知评分的平均值（代表大环境基准）
        self.mu = np.mean(R[R > 0]) if np.any(R > 0) else 0

        # B. 用户偏置 (m x 1)：提取用户个人的打分习惯（手松或手紧）
        self.bu = np.zeros(self.m)

        # C. 物品偏置 (1 x n)：提取电影本身的受喜爱程度（神作或烂片）
        self.bi = np.zeros(self.n)

        # D. 隐因子矩阵：用于提取核心特征
        # P 矩阵 (m x k): 用户对特征的偏好
        self.P = np.random.normal(scale=1./k, size=(self.m, self.k))
        # Q 矩阵 (n x k): 电影拥有的特征
        self.Q = np.random.normal(scale=1./k, size=(self.n, self.k))

    def train(self, epochs=300):
        """
        通过迭代优化，将观测矩阵分解为：R ≈ mu + bu + bi + P·Q^T
        """
        history = []
        for epoch in range(epochs):
            for u in range(self.m):
                for i in range(self.n):
                    if self.R[u, i] > 0:
                        # 1. 计算预测值 (生成过程)
                        pred = self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
                        # 2. 计算误差 (评价过程)
                        error = self.R[u, i] - pred

                        # 3. 更新偏置项 (降噪：剥离非特征因素)
                        self.bu[u] += self.lr * (error - self.lam * self.bu[u])
                        self.bi[i] += self.lr * (error - self.lam * self.bi[i])

                        # 4. 更新隐因子矩阵 (特征提取)
                        p_old = self.P[u].copy()
                        self.P[u] += self.lr * (error * self.Q[i] - self.lam * self.P[u])
                        self.Q[i] += self.lr * (error * p_old - self.lam * self.Q[i])

            # 评价指标：计算均方根误差 RMSE
            if (epoch + 1) % 50 == 0:
                mse = np.mean([(self.R[u,i] - (self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])))**2
                               for u in range(self.m) for i in range(self.n) if self.R[u,i] > 0])
                rmse = np.sqrt(mse)
                history.append(rmse)
                print(f"Epoch {epoch+1}: RMSE = {rmse:.4f}")
        return history

    def predict(self, u_idx, i_idx):
        # 最终预测分 = 全局均值 + 用户偏置 + 物品偏置 + 特征匹配点积
        return self.mu + self.bu[u_idx] + self.bi[i_idx] + np.dot(self.P[u_idx], self.Q[i_idx])

# 实例化并训练
model = BiasSVDRecommender(R)
model.train()

# ==========================================
# 3. 接口层
# ==========================================

@app.get("/recommend/{username}")
def get_recommendation(username: str):
    if username not in USER_NAMES: return {"error": "User missing"}
    u_idx = USER_NAMES.index(username)

    recs = []
    for i_idx, m_name in enumerate(MOVIE_NAMES):
        score = model.predict(u_idx, i_idx)
        recs.append({
            "movie": m_name,
            "score": round(float(score), 2),
            "type": MOVIE_TAGS[m_name]
        })

    recs.sort(key=lambda x: x["score"], reverse=True)
    return {"user": username, "recommendations": recs}

@app.get("/visualize")
def visualize():
    """ 可视化隐空间：展示矩阵分解后的特征提取结果 """
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # 绘制用户特征向量 (P)
    for i, name in enumerate(USER_NAMES):
        plt.scatter(model.P[i, 0], model.P[i, 1], c='red', s=100)
        plt.text(model.P[i, 0]+0.01, model.P[i, 1]+0.01, f"用户:{name}")

    # 绘制电影特征向量 (Q)
    for j, title in enumerate(MOVIE_NAMES):
        plt.scatter(model.Q[j, 0], model.Q[j, 1], c='blue', marker='x', s=100)
        plt.text(model.Q[j, 0]+0.01, model.Q[j, 1]+0.01, f"电影:{title}")

    plt.title("BiasSVD 隐空间特征可视化 (k=2)")
    plt.xlabel("隐因子 1 (例如: 动作程度)")
    plt.ylabel("隐因子 2 (例如: 情感深度)")
    plt.grid(True, alpha=0.2)

    # 支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
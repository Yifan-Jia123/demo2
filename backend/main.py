from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 接收前端传来的自定义打分格式
class UserRatings(BaseModel):
    ratings: Dict[str, Optional[float]]


# 1. 模拟数据库中其他用户的历史打分 (让矩阵不至于太空，方便 SVD 找规律)
# 顺序: ["复仇者联盟", "蜘蛛侠", "泰坦尼克号", "爱情故事"]
historical_data = [
    [5.0, 4.0, np.nan, np.nan],  # 类似小明（爱科幻）
    [np.nan, np.nan, 4.0, 5.0],  # 类似小红（爱爱情）
    [5.0, np.nan, np.nan, 4.0],  # 类似小刚（都爱）
    [4.0, 5.0, 1.0, 2.0],  # 路人甲
    [1.0, 2.0, 5.0, 5.0],  # 路人乙
]
movie_names = ["复仇者联盟", "蜘蛛侠", "泰坦尼克号", "爱情故事"]


@app.post("/recommend")
def get_recommendation(user_input: UserRatings):
    # 2. 提取当前测试用户的打分，并加入矩阵
    new_user_row = []
    for m in movie_names:
        val = user_input.ratings.get(m)
        new_user_row.append(val if val is not None else np.nan)

    # 构建完整的 M 矩阵 (历史用户 + 当前新用户)
    full_matrix = historical_data + [new_user_row]
    M = np.array(full_matrix, dtype=float)

    # 3. 处理缺失值：SVD 不接受 NaN，我们用该电影的平均分来填充空缺
    col_mean = np.nanmean(M, axis=0)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(col_mean, inds[1])

    # 4. 核心：执行 SVD 奇异值分解
    # U: 用户特征矩阵, sigma: 奇异值(权重), Vt: 物品特征矩阵
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)

    # 降维：我们假设有 2 个隐藏特征（科幻、爱情）
    k = 2
    U_k = U[:, :k]
    sigma_k = np.diag(sigma[:k])
    Vt_k = Vt[:k, :]

    # 5. 重建矩阵，得到所有空缺位置的预测打分
    # 公式：M' ≈ U * Σ * V^T
    M_reconstructed = np.dot(U_k, np.dot(sigma_k, Vt_k))

    # 提取最后一行，这就是我们刚刚输入的“当前用户”的预测分数
    current_user_predictions = M_reconstructed[-1]

    # 6. 整理预测结果并生成推荐
    predictions_dict = {}
    recommendation = None
    max_score = -float('inf')

    for i, movie in enumerate(movie_names):
        # 限制分数在 0-5 范围内，看起来更真实
        score = min(max(float(current_user_predictions[i]), 0), 5)
        predictions_dict[movie] = round(score, 2)

        # 只推荐用户【没看过】（原本为 NaN）的电影
        if np.isnan(new_user_row[i]):
            if score > max_score:
                max_score = score
                recommendation = movie

    # 如果全看过了，就挑预测分最高的
    if recommendation is None:
        recommendation = max(predictions_dict, key=predictions_dict.get)

    return {
        "predictions": predictions_dict,
        "recommendation": recommendation
    }


if __name__ == "__main__":
    import uvicorn

    # 如果要在局域网展示，保持 0.0.0.0；如果只是自己测，改回 127.0.0.1 也行
    uvicorn.run(app, host="127.0.0.1", port=8000)
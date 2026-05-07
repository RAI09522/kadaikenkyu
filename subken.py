import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 基本パラメータ設定
# ==========================================
N_AGENTS = 100        # エージェント数（計算負荷を下げるため100）
T_STEPS = 100         # 通常シミュレーションのステップ数
T_SHOCK = 50          # アルゴリズム・ショックが発生するステップ

PHI = 0.1             # \phi: 忘却率
KAPPA = 5.0           # \kappa: 露出変換係数
BETA_NET = 0.5        # \beta_{net}: ネットワーク効果
SIGMA = 0.05          # \sigma: ボラティリティ

ALPHA_ALG = 1.5       # \alpha: 初期アルゴリズムの可視性弾力性
BETA_ALG = 0.8        # \beta: 初期アルゴリズムの品質弾力性

LAMBDA_V = 2.0        # \lambda: 認知バイアス
V_BAR = 5.0           # \bar{V}: バズりの閾値
GAMMA_V = 1.2         # \gamma: 可視性の基本需要弾力性
MAX_ATTENTION = 5000  # 注意保存則

C_Q_COEF = 0.5        # 品質投資コスト係数
C_E_COEF = 0.2        # エフォート投資コスト係数
LEARNING_RATE = 0.05  # エージェントの学習率

# ==========================================
# モジュール1: マクロ市場の動学計算関数
# ==========================================
def run_market_step(V, Q, P, e, alpha, beta, phi):
    V_safe = np.clip(V, 0.01, None)
    
    # [Eq. 4] 露出シェア E_i
    scores = (V_safe ** alpha) * (Q ** beta)
    E = scores / np.sum(scores)
    
    # [Eq. 3] 需要関数 D_i
    Phi_V = (V_safe ** GAMMA_V) / (1 + np.exp(-LAMBDA_V * (V_safe - V_BAR)))
    Pot_Demand = Phi_V * (Q / P)
    
    # [Eq. 7] 注意保存則
    Total_Demand = np.sum(Pot_Demand)
    if Total_Demand > MAX_ATTENTION:
        D = Pot_Demand * (MAX_ATTENTION / Total_Demand)
    else:
        D = Pot_Demand
        
    # [Eq. 6] 利益 \Pi_i
    Cost_Q = C_Q_COEF * (Q ** 2)
    Cost_e = C_E_COEF * (e ** 2)
    Profit = P * D - Cost_Q - Cost_e
    
    # [Eq. 5] 可視資本の動学 dV
    Alg_Effect = KAPPA * E * e
    Net_Effect = BETA_NET * (V_safe ** 0.5) * E
    Forget_Effect = phi * V_safe
    Stochastic_Shock = SIGMA * V_safe * np.random.randn(N_AGENTS)
    
    dV = Alg_Effect + Net_Effect - Forget_Effect + Stochastic_Shock
    V_new = np.clip(V + dV, 0.01, None)
    
    return V_new, Profit

# ジニ係数計算
def calc_gini(x):
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

# ==========================================
# モジュール2: ① 感度分析（パラメータ・スウィープ）
# ==========================================
def simulate_parameter_sweep():
    alpha_range = np.linspace(0.0, 3.0, 20)
    final_ginis = []
    
    np.random.seed(42)
    Q = np.random.uniform(1, 10, N_AGENTS)
    P = np.random.uniform(10, 100, N_AGENTS)
    e = np.random.uniform(0.1, 2.0, N_AGENTS)
    
    for test_alpha in alpha_range:
        V_test = np.random.uniform(1, 2, N_AGENTS)
        for _ in range(50): # 各アルファで50ステップ回して定常状態を見る
            V_test, _ = run_market_step(V_test, Q, P, e, test_alpha, BETA_ALG, PHI)
        final_ginis.append(calc_gini(V_test))
        
    return alpha_range, final_ginis

# ==========================================
# モジュール3: ② ショック実験 ＆ ③ エージェント学習
# ==========================================
def simulate_shock_and_learning():
    np.random.seed(42)
    Q = np.random.uniform(1, 10, N_AGENTS)
    P = np.random.uniform(10, 100, N_AGENTS)
    e = np.random.uniform(0.1, 2.0, N_AGENTS)
    V = np.random.uniform(1, 2, N_AGENTS)
    
    hist_V = np.zeros((T_STEPS, N_AGENTS))
    hist_e = np.zeros((T_STEPS, N_AGENTS))
    prev_Profit = np.zeros(N_AGENTS)
    prev_e = np.copy(e)
    
    current_alpha = ALPHA_ALG
    current_phi = PHI
    
    for t in range(T_STEPS):
        # --- アルゴリズム・ショックの発生 ---
        if t == T_SHOCK:
            current_alpha = 0.5 # 可視性偏重をやめ、品質重視に変更
            current_phi = 0.8   # 過去の可視性をほぼリセット（忘却率激増）
        elif t == T_SHOCK + 5:
            current_phi = PHI   # 忘却率は元に戻る
            
        V_new, Profit = run_market_step(V, Q, P, e, current_alpha, BETA_ALG, current_phi)
        
        # --- エージェントの自律学習（勾配法に基づくエフォート e の調整） ---
        if t > 0:
            delta_Profit = Profit - prev_Profit
            delta_e = e - prev_e
            # 利益が増えたなら同じ方向へ、減ったなら逆方向へ e を調整
            direction = np.sign(delta_Profit * delta_e + 1e-5) 
            e_new = e + LEARNING_RATE * direction * e
            e_new = np.clip(e_new, 0.1, 5.0) # エフォートの限界値
        else:
            e_new = np.copy(e)
            
        # 状態の更新
        prev_Profit = Profit
        prev_e = np.copy(e)
        e = e_new
        V = V_new
        
        hist_V[t, :] = V
        hist_e[t, :] = e
        
    return hist_V, hist_e, Q

# ==========================================
# 実行と可視化
# ==========================================
alpha_range, ginis = simulate_parameter_sweep()
hist_V, hist_e, Q_agents = simulate_shock_and_learning()

plt.figure(figsize=(15, 5))

# グラフ1: 感度分析（相転移）
plt.subplot(1, 3, 1)
plt.plot(alpha_range, ginis, marker='o', color='purple')
plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Phase Transition')
plt.title("Phase Transition of Market Inequality")
plt.xlabel(r"Algorithm's Visibility Elasticity ($\alpha$)")
plt.ylabel("Gini Coefficient of $V$")
plt.grid(True, alpha=0.3)
plt.legend()

# グラフ2: ショック実験と可視資本の動学
plt.subplot(1, 3, 2)
for i in range(N_AGENTS):
    color = 'blue' if Q_agents[i] > np.percentile(Q_agents, 90) else 'red' if Q_agents[i] < np.percentile(Q_agents, 10) else 'gray'
    alpha_val = 0.6 if color != 'gray' else 0.1
    plt.plot(hist_V[:, i], color=color, alpha=alpha_val)
plt.axvline(x=T_SHOCK, color='black', linestyle='--', linewidth=2, label='Algorithm Shock')
plt.title("Visibility Dynamics with Algorithm Shock")
plt.xlabel("Time Step")
plt.ylabel("Visibility Capital ($V$)")
plt.yscale('log') # 差を見やすくするため対数スケール
plt.legend(['Top 10% Q', 'Bottom 10% Q', 'Others', 'Shock'][3:], loc='upper right')

# グラフ3: エージェントの学習過程（エフォート投資の推移）
plt.subplot(1, 3, 3)
plt.plot(np.mean(hist_e, axis=1), color='green', linewidth=2, label='Average Effort')
plt.axvline(x=T_SHOCK, color='black', linestyle='--')
plt.title("Agent Learning: Average Effort ($e$) over time")
plt.xlabel("Time Step")
plt.ylabel("Investment in Algorithm SEO ($e$)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

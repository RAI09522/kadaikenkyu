import numpy as np

def run_investment_experiment():
    print("=== 投資戦略シミュレーション実行中 ===\n")

    # --- 共通パラメータの設定 ---
    base_amount = 100000       # 基本積立額 (B)
    p_avg = 950                # 自分の平均取得単価
    current_price = 850        # 現在の株価 (P(t))
    prev_price = 880           # 前日の終値 (パニック判定用)
    
    # 市場データ (S&P500とNVIDIAのボラティリティ)
    market_vol = 0.18          # 米国市場平均ボラ
    nvda_vol = 0.45            # NVIDIAの現在のボラ
    
    # 企業分析データ (IR指標)
    pe_current = 30            # 現在のPER
    pe_target = 35             # 基準PER
    pbr_current = 12           # 現在のPBR
    pbr_target = 15            # 基準PBR
    
    # テクニカル指標
    sma200 = 800               # 200日移動平均線
    loss_duration = 50         # 含み損継続日数

    # ---------------------------------------------------------
    # 1. 損切り・生存判定アルゴリズム (S(t))
    # ---------------------------------------------------------
    # すべての条件をパスすれば 1 (継続)、一つでも破れば 0 (撤退)
    f1 = 1 if current_price > (sma200 * 0.7) else 0      # トレンド維持
    f2 = 1 if current_price > (prev_price * 0.8) else 0   # 異常急落ガード
    f3 = 1 if loss_duration < 180 else 0                 # 時間制限
    
    survival_signal = f1 * f2 * f3  # 損切り判定
    omega = 1 if current_price > (sma200 * 0.8) else 0   # 生存判定

    # ---------------------------------------------------------
    # 2. DVA-AAMロジックの計算
    # ---------------------------------------------------------
    # 安全装置 (Phi)
    target_vol = market_vol * 1.5
    phi = max(0, 1 - (nvda_vol - target_vol) / 10)
    
    # 動的ラムダ (加点ロジック)
    # base(2.0) * PER割安度 * (1 + PBR割安度)
    dynamic_lambda = 2.0 * (pe_target / pe_current) * (1 + (pbr_target / pbr_current))
    
    # 加速装置 (Psi): ネイピア数 e を使用
    # 下落率 = (平均単価 - 現在価格) / 平均単価
    price_gap_ratio = (p_avg - current_price) / p_avg
    psi = np.exp(dynamic_lambda * price_gap_ratio)
    
    # 【最終結果】DVA-AAM投資額
    dva_amount = base_amount * phi * psi * omega * survival_signal

    # ---------------------------------------------------------
    # 3. ドルコスト平均法 (DCA)
    # ---------------------------------------------------------
    # 常に一定額。ただし、生存判定や損切り条件は共通とする
    dca_amount = base_amount * survival_signal

    # ---------------------------------------------------------
    # 結果の表示
    # ---------------------------------------------------------
    print(f"【判定結果】")
    print(f"  - 損切り判定: {'継続' if survival_signal == 1 else '強制売却・撤退'}")
    print(f"  - 生存判定  : {'生存' if omega == 1 else '投資停止'}")
    print(f"  - 動的ラムダ: {dynamic_lambda:.2f}")
    print("-" * 30)
    print(f"【投資額の比較】")
    print(f"  1. ドルコスト平均法: {dca_amount:,.0f} 円")
    print(f"  2. DVA-AAMモデル   : {dva_amount:,.0f} 円")
    print("-" * 30)
    
    if dva_amount > dca_amount:
        diff = (dva_amount / dca_amount - 1) * 100
        print(f">> DVA-AAMはドルコストより {diff:.1f}% 多く投資を推奨しています（チャンス！）")
    elif dva_amount == 0 and survival_signal == 1:
         print(">> ボラティリティ過大、またはトレンド割れにより待機中です。")

# 実行
if __name__ == "__main__":
    run_investment_experiment()
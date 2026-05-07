import requests
import pandas as pd
import time

def generate_steam_csv(app_ids, filename='steam_market_data.csv'):
    data_list = []
    print(f"全 {len(app_ids)} 件のデータ取得を開始します...")
    
    for i, app_id in enumerate(app_ids):
        # 1. Steam Store API から価格情報を取得
        store_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        
        # 2. SteamSpy API から需要(D)と可視性(V)を取得
        spy_url = f"https://steamspy.com/api.php?request=appdetails&appid={app_id}"
        
        try:
            store_response = requests.get(store_url).json()
            spy_response = requests.get(spy_url).json()
            
            if store_response and str(app_id) in store_response and store_response[str(app_id)]['success']:
                app_data = store_response[str(app_id)]['data']
                
                # 価格 (P): 無料ゲームは0、それ以外はドル換算など
                price = app_data['price_overview']['initial'] / 100 if 'price_overview' in app_data else 0
                
                # 品質 (Q): (好評数 / 総レビュー数) * 10
                pos_reviews = spy_response.get('positive', 0)
                neg_reviews = spy_response.get('negative', 0)
                total_reviews = pos_reviews + neg_reviews
                quality_score = (pos_reviews / total_reviews * 10) if total_reviews > 0 else 0
                
                # 可視性 (V): フォロワー数
                visibility = spy_response.get('followers', 0)
                
                # 需要 (D): 推定所有者数の中央値
                owners_str = spy_response.get('owners', '0 .. 0')
                owners_bounds = [int(x.replace(',', '')) for x in owners_str.split(' .. ')]
                demand = sum(owners_bounds) / 2
                
                data_list.append({
                    'AppID': app_id,
                    'Name': app_data.get('name', 'Unknown'),
                    'Price_P': price,
                    'Quality_Q': quality_score,
                    'Visibility_V': visibility,
                    'Demand_D': demand
                })
                print(f"[{i+1}/{len(app_ids)}] 取得成功: {app_data.get('name')}")
        except Exception as e:
            print(f"[{i+1}/{len(app_ids)}] エラー (AppID {app_id}): {e}")
            
        # APIサーバーに負荷をかけないための待機（超重要）
        time.sleep(1.5)
        
    # データフレームに変換してCSVとして保存
    df = pd.DataFrame(data_list)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n完了！ {filename} を保存しました。")

# ==========================================
# 取得するゲームのAppIDリスト（ここでサンプル数を決めます）
# ※本格的な研究時は、Steam APIの「GetAppList」等で数千件のIDを一括取得してここに渡します。
# 今回はテスト用に有名タイトルとインディーゲームを混ぜた20件のリストを用意しました。
# ==========================================
sample_ids = [
    730, 271590, 1145360, 1293830, 105600, 413150, 252490, 322330, 
    292030, 391540, 289070, 1086940, 892970, 1172470, 646570, 
    1151640, 1326470, 553850, 230410, 236850
]

# スクリプト実行
generate_steam_csv(sample_ids)

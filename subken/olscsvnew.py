import requests
import pandas as pd
import time

def generate_steam_csv(app_ids, filename='steam_market_data.csv'):
    data_list = []

    print(f"全 {len(app_ids)} 件のデータ取得を開始します...")

    for i, app_id in enumerate(app_ids):

        store_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        spy_url = f"https://steamspy.com/api.php?request=appdetails&appid={app_id}"

        try:
            # =========================
            # API取得
            # =========================
            store_response = requests.get(store_url, timeout=10).json()
            spy_response = requests.get(spy_url, timeout=10).json()

            if (
                store_response
                and str(app_id) in store_response
                and store_response[str(app_id)]['success']
            ):

                app_data = store_response[str(app_id)]['data']

                # =========================
                # 価格
                # =========================
                if 'price_overview' in app_data:
                    price = app_data['price_overview']['initial'] / 100
                else:
                    price = 0

                # =========================
                # 品質
                # =========================
                pos_reviews = spy_response.get('positive', 0)
                neg_reviews = spy_response.get('negative', 0)

                total_reviews = pos_reviews + neg_reviews

                if total_reviews > 0:
                    quality_score = (pos_reviews / total_reviews) * 10
                else:
                    quality_score = 0

                # =========================
                # 可視性
                # =========================
                visibility = spy_response.get('followers', 0)

                # =========================
                # 需要
                # =========================
                owners_str = spy_response.get('owners', '0 .. 0')

                try:
                    owners_bounds = [
                        int(x.replace(',', '').strip())
                        for x in owners_str.split(' .. ')
                    ]

                    demand = sum(owners_bounds) / 2

                except:
                    demand = 0

                # =========================
                # 保存
                # =========================
                data_list.append({
                    'AppID': app_id,
                    'Name': app_data.get('name', 'Unknown'),
                    'Price_P': price,
                    'Quality_Q': quality_score,
                    'Visibility_V': visibility,
                    'Demand_D': demand
                })

                print(
                    f"[{i+1}/{len(app_ids)}] "
                    f"成功: {app_data.get('name')} | "
                    f"P={price} Q={quality_score:.2f} "
                    f"V={visibility} D={demand}"
                )

            else:
                print(f"[{i+1}/{len(app_ids)}] Store API失敗")

        except Exception as e:
            print(f"[{i+1}/{len(app_ids)}] エラー: {e}")

        # API負荷対策
        time.sleep(3)

    # =========================
    # CSV保存
    # =========================
    df = pd.DataFrame(data_list)

    print("\n===== 取得データ確認 =====")
    print(df.head())

    print("\n===== 行数 =====")
    print(len(df))

    df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"\n完了: {filename} を保存しました")


# ==========================================
# サンプルAppID
# ==========================================
sample_ids = [
    730,
    271590,
    1145360,
    1293830,
    105600,
    413150,
    252490,
    322330,
    292030,
    391540,
    289070,
    1086940,
    892970,
    1172470,
    646570,
    1151640,
    1326470,
    553850,
    230410,
    236850
]

# 実行
generate_steam_csv(sample_ids)
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# =====================================================================
# ⚠️ CRITICAL REQUIREMENT: DATA PATH PLACEHOLDER
# 請在此處輸入您的資料檔案路徑 (支援 .csv 或 .xlsx / .xls)
# 建議您填入剛才產出的 engineered_stock_prices.csv
# =====================================================================
DATA_PATH = "請在此輸入您的資料檔案路徑（如 .csv 或 .xlsx）"


class FeatureScaler:
    """
    確保不同量級（Scale）的特徵在模型中能夠「公平競爭」，
    提供標準化（Z-score）與最小最大縮放（Min-Max）兩種核心策略。
    """

    def __init__(self):
        pass

    def scale_by_standardization(
        self, df: pd.DataFrame, cols: list
    ) -> pd.DataFrame:
        """策略一：標準化縮放 (Standardization / Z-score Scaling)

        【數學原理】
        公式：$Z = \frac{X - \mu}{\sigma}$
        將原始資料減去該特徵的平均數（$\mu$），再除以標準差（$\sigma$）。
        轉換後，資料的分佈將會被平移並縮放至「平均數為 0，標準差為 1」的狀態。

        【統計考量與直覺本質】
        - 中心化 (Zero-centered)：資料的重心被定錨在原點 (0)。大於平均的值為正，小於平均的值為負，
          這對很多利用向量內積（如 SVM、神經網路）的演算法有極大幫助，能加快收斂速度。
        - 較強的穩健性 (Robustness)：相較於 MinMax，標準化並沒有把特徵硬塞進一個死板的邊界內。
          如果資料中有極端值，它雖然會稍微拉扯平均數與標準差，但絕大多數的正常資料點依然會
          保有一段合理的相對距離，不至於全軍覆沒。這也是為什麼多數基於梯度下降（Gradient Descent）
          或距離計算的機器學習演算法（如邏輯回歸、KNN、SVM）都強烈偏好此方法。

        Parameters:
        -----------
        df : pd.DataFrame
            輸入的資料集。
        cols : list
            需要進行標準化縮放的數值欄位清單。

        Returns:
        --------
        pd.DataFrame
            完成標準化縮放的資料集。
        """
        df_scaled = df.copy()
        existing_cols = [col for col in cols if col in df_scaled.columns]

        if not existing_cols:
            print("[Warning] 未找到指定的縮放欄位，已略過標準化。")
            return df_scaled

        scaler = StandardScaler()
        # 進行擬合與轉換 (Fit and Transform)
        df_scaled[existing_cols] = scaler.fit_transform(df_scaled[existing_cols])
        print(f"[Scaling: 標準化] 成功將欄位 {existing_cols} 轉換為平均值 0、標準差 1 的分佈。")

        return df_scaled

    def scale_by_minmax(
        self, df: pd.DataFrame, cols: list
    ) -> pd.DataFrame:
        """策略二：最小最大縮放 (Min-Max Scaling)

        【數學原理】
        公式：$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$
        找出特徵的最大值與最小值，以此作為尺規，將所有數據等比例線性壓縮到固定的 $[0, 1]$ 區間內。

        【統計考量與直覺本質】
        - 映射資料邊界：直覺來說，班上最低分就是 0，最高分就是 1，其餘的人根據比例拿對應的相對分數。
          這在需要輸出機率或特定分數權重的場合非常實用。
        - ⚠️ 致命缺點 (極端值的暴政)：如果這個特徵有極端值（例如：多數人的年收入在 50~100 萬之間，
          但資料中突然出現一個郭董年收入 100 億）。此時 $X_{max}$ 變成 100 億，這會導致所有正常的上班族
          被強行壓縮在 $0$ 到 $0.0001$ 這個肉眼與模型都無法分辨的極小區間內，該特徵將完全失去分辨力。

        【適用情境】
        適用於「已知明確上下界且無極端值」的資料，例如：
        影像處理的像素範圍（恆為 0 到 255）、RGB 色碼，或有明確滿分的考試評分系統。

        Parameters:
        -----------
        df : pd.DataFrame
            輸入的資料集。
        cols : list
            需要進行 Min-Max 縮放的數值欄位清單。

        Returns:
        --------
        pd.DataFrame
            完成最小最大縮放的資料集。
        """
        df_scaled = df.copy()
        existing_cols = [col for col in cols if col in df_scaled.columns]

        if not existing_cols:
            print("[Warning] 未找到指定的縮放欄位，已略過 Min-Max 縮放。")
            return df_scaled

        scaler = MinMaxScaler()
        # 進行擬合與轉換 (Fit and Transform)
        df_scaled[existing_cols] = scaler.fit_transform(df_scaled[existing_cols])
        print(f"[Scaling: Min-Max] 成功將欄位 {existing_cols} 壓縮至 [0, 1] 區間。")

        return df_scaled


# =====================================================================
# MAIN 執行進入點 (比較與對照展示)
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 啟動特徵縮放 (Feature Scaling) 分析模組")
    print("=" * 60)

    # 1. 驗證資料路徑，若無檔案則建立展示專用的極端值數據
    if not DATA_PATH.strip() or "請在此輸入" in DATA_PATH or not os.path.exists(DATA_PATH):
        print(f"[Info] 找不到實體檔案，或使用了預設路徑。")
        print("[Info] 系統自動建立包含「極端值」的 Dummy 資料集來展示兩種方法的致命差異...\n")
        
        # 故意設計一個帶有極端值 (Volume = 10,000,000) 的資料
        dummy_data = {
            "Ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "Close": [150.0, 152.0, 149.0, 155.0, 151.0],         # 正常分佈
            "Volume": [2000.0, 2500.0, 1800.0, 10000000.0, 2200.0] # 第4筆為極端異常值
        }
        df_raw = pd.DataFrame(dummy_data)
    else:
        # 讀取真實資料
        try:
            df_raw = pd.read_csv(DATA_PATH) if DATA_PATH.endswith(".csv") else pd.read_excel(DATA_PATH)
            print("📋 成功載入目標資料集！")
        except Exception as e:
            print(f"💥 讀取失敗: {e}")
            exit()

    # 定義需要縮放的數值欄位
    # 💡 夥伴注意：請確認您的真實資料中有 Close 與 Volume 這兩個欄位
    target_numeric_cols = ["Close", "Volume"]
    
    # 確保欄位皆為數值且無缺失值 (補 0 作為示範防護)
    for col in target_numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

    # 2. 實例化特徵縮放器
    scaler_agent = FeatureScaler()

    # 3. 執行兩種縮放策略
    df_standardized = scaler_agent.scale_by_standardization(df_raw, cols=target_numeric_cols)
    df_minmax = scaler_agent.scale_by_minmax(df_raw, cols=target_numeric_cols)

    # 4. 輸出對比報告
    print("\n" + "=" * 60)
    print("📊 特徵縮放前後對比報告 (Scaling Comparison Report)")
    print("=" * 60)

    for col in target_numeric_cols:
        if col not in df_raw.columns:
            continue
            
        print(f"🎯 欄位: {col}")
        
        # 提取原始、標準化、MinMax 的數據序列
        raw_series = df_raw[col]
        std_series = df_standardized[col]
        mm_series = df_minmax[col]
        
        # 建立對比 DataFrame
        compare_df = pd.DataFrame({
            "方法 (Method)": ["原始資料 (Raw)", "標準化 (Standardization)", "MinMax [0,1]"],
            "最小值 (Min)": [raw_series.min(), std_series.min(), mm_series.min()],
            "最大值 (Max)": [raw_series.max(), std_series.max(), mm_series.max()],
            "平均值 (Mean)": [raw_series.mean(), std_series.mean(), mm_series.mean()],
            "標準差 (Std)": [raw_series.std(), std_series.std(), mm_series.std()]
        })
        
        # 格式化輸出以提高可讀性
        print(compare_df.to_string(index=False, float_format="{:12.4f}".format))
        print("-" * 60)

    print("\n💡 專家洞察 (Expert Insight):")
    print("👉 請觀察有極端值的欄位 (如 Volume)。在 MinMax 縮放中，最大值變成了 1.0000，")
    print("   但平均值與標準差被極端值擠壓到了幾乎接近 0 的水準，這會導致模型無法區分多數正常資料。")
    print("   相反地，標準化 (Standardization) 雖然將最大值拉到了非常高的正數 (Z-score 很大)，")
    print("   但依然完美維持了整體分佈的平均值為 0.0000，標準差為 1.0000，保留了資料的抵抗力！")
    print("=" * 60)
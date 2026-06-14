import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# =====================================================================
# ⚠️ CRITICAL REQUIREMENT: DATA PATH PLACEHOLDER
# 請在此處輸入您的資料檔案路徑 (支援 .csv 或 .xlsx / .xls)
# =====================================================================
DATA_PATH = ""


class DataAuditor:
    """
    具備重複值去重、Z-Score 檢驗、IQR 四分位距法及 Isolation Forest 機器學習異常偵測功能。
    """

    def __init__(self):
        pass

    def detect_and_drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """策略一：重複值去重 (Duplicate Detection and Removal)

        【最適合處理的資料情境】
        任何資料清洗流程的第一道防線。最常用於網路爬蟲重複抓取、系統伺服器重疊寫入、
        或是多個資料流管道（Data Pipelines）合併時產生的「鏡像重複」數據。

        【統計原理與直覺本質】
        - 避免權重被無端放大：想像一架天平，如果相同的資料不小心被「倒了兩次」，天平的重心就會嚴重向該樣本傾斜。
          在統計學或機器學習中，重複的鏡像觀測值會被人為地放大權重，導致平均數偏移、變異數被低估，
          模型甚至會去「死記硬背」這些重複的樣本（導致嚴重的過擬合 Overfitting）。
          剔除重複值，就是還原數據集最真實的隨機抽樣結構。

        Parameters:
        -----------
        df : pd.DataFrame
            輸入的原始資料集。

        Returns:
        --------
        pd.DataFrame
            去重後的資料集。
        """
        df_audited = df.copy()
        duplicate_count = df_audited.duplicated().sum()

        print(f"[Audit: 重複值] 偵測到重複列數: {duplicate_count} 筆")
        if duplicate_count > 0:
            df_audited = df_audited.drop_duplicates()
            print(f"[Audit: 重複值] 已成功剔除重複資料，剩餘列數: {df_audited.shape[0]} 筆")

        return df_audited

    def detect_outliers_zscore(
        self, df: pd.DataFrame, threshold: float = 3.0
    ) -> pd.DataFrame:
        """策略二：統計學標準差法 (Z-Score Outlier Detection)

        【最適合處理的資料情境】
        適用於已知或高度懷疑該數據分佈符合「常態分佈（Normal Distribution）」的欄位（如：成人的身高、正常製造業的零件誤差）。

        【統計原理與直覺本質】
        - 常態分佈的邊界守護者：大自然與多數人類經濟行為在大量隨機抽樣下，皆會收斂成鐘形曲線（常態分佈）。
          依據統計學著名的「68-95-99.7 法則」，有 99.7% 的資料點會落在距離平均數正負 3 個標準差的區間內。
          Z-Score 的本質，就是計算「該資料點距離平均數有幾個標準差」。
          如果 $|Z| > 3$，意味著這個事件在常態世界中發生的機率小於 0.3%，屬於「極端罕見的特例」。
          在計量經濟學中，這通常代表著資料錄入錯誤（如打錯小數點）或是遭逢了黑天鵝極端事件。

        Parameters:
        -----------
        df : pd.DataFrame
            輸入的資料集。
        threshold : float, default 3.0
            標準差臨界值，通常設定為 3.0。

        Returns:
        --------
        pd.DataFrame
            新增 'is_outlier_zscore' 欄位（布林值）的 DataFrame。
        """
        df_res = df.copy()
        numeric_cols = df_res.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.Series(False, index=df_res.index)

        for col in numeric_cols:
            col_mean = df_res[col].mean()
            col_std = df_res[col].std()

            if col_std > 0:
                # 計算 Z-Score
                z_scores = (df_res[col] - col_mean) / col_std
                col_outliers = z_scores.abs() > threshold
                outlier_mask = outlier_mask | col_outliers

        df_res["is_outlier_zscore"] = outlier_mask
        print(f"[Audit: Z-Score] 偵測到異常值數量: {outlier_mask.sum()} 筆 (臨界值: > {threshold} 標準差)")
        return df_res

    def detect_outliers_iqr(
        self, df: pd.DataFrame, factor: float = 1.5
    ) -> pd.DataFrame:
        """策略三：四分位距法 (IQR Outlier Detection / Tukey's Fences)

        【最適合處理的資料情境】
        極度適合處理「非正常分佈」、「嚴重偏態（Skewed）」或含有大量極端值的情境，
        例如：金融市場的股票成交量、房地產價格、人民年收入等。

        【統計原理與直覺本質】
        - 強大的無母數防禦力：Z-Score 雖然好用，但它致命的弱點在於「平均數與標準差極易受到極端值的拉扯」。
          如果資料中本來就有一個巨大的異常值，標準差會被無端放大，導致 Z-Score「防禦失效」。
          四分位距（IQR = Q3 - Q1）則是利用資料的排位（第 25% 與第 75% 的切點）。
          無論極端值有多大，都不會影響 Q1 與 Q3 的位置，因此 IQR 具備極強的穩健性（Robustness）。
          以 $[Q1 - 1.5 \\times IQR, Q3 + 1.5 \\times IQR]$ 作為邊界，
          就像是在資料最密集的「中產階級核心區」向外築起兩道圍牆，能極其敏銳且不帶分佈假設地抓出真正的邊緣異常者。

        Parameters:
        -----------
        df : pd.DataFrame
            輸入的資料集。
        factor : float, default 1.5
            四分位距的乘數因子，1.5 判定為異常值（Outlier），3.0 則通常判定為極端異常值（Extreme Outlier）。

        Returns:
        --------
        pd.DataFrame
            新增 'is_outlier_iqr' 欄位（布林值）的 DataFrame。
        """
        df_res = df.copy()
        numeric_cols = df_res.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.Series(False, index=df_res.index)

        for col in numeric_cols:
            q1 = df_res[col].quantile(0.25)
            q3 = df_res[col].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - (factor * iqr)
            upper_bound = q3 + (factor * iqr)

            col_outliers = (df_res[col] < lower_bound) | (df_res[col] > upper_bound)
            outlier_mask = outlier_mask | col_outliers

        df_res["is_outlier_iqr"] = outlier_mask
        print(f"[Audit: IQR 法] 偵測到異常值數量: {outlier_mask.sum()} 筆 (邊界因子: {factor})")
        return df_res

    def detect_outliers_isolation_forest(
        self, df: pd.DataFrame, contamination: float = 0.05
    ) -> pd.DataFrame:
        """策略四：機器學習演算法 (Isolation Forest)

        【最適合處理的資料情境】
        適用於「高維度欄位」且欄位間具備複雜的「交叉複合異常」情境。
        例如：信用卡盜刷（單看金額正常，單看時間也正常，但深夜在國外連續大筆刷卡就是異常）。

        【統計原理與直覺本質】
        - 孤立不合群的邊緣人：大部分的異常偵測思維是「先描繪出正常人的樣貌（建構密集區），再把外面的人當異常」。
          隔離森林（Isolation Forest）採取了完全反向的操作：它直接玩「隨機切分遊戲」。
          如果我們對空間中的資料隨機畫切刀（隨割二元樹），那些聚集成群的「正常資料」，需要切非常多刀才能被單獨分離出來；
          相反地，那些特徵怪異、不合群的「異常觀測值」，因為在空間中結構孤立，只要簡短的幾刀（淺層路徑）就會被瞬間切分出來。
          這種利用「路徑長短」來衡量孤立程度的邏輯，速度極快，且完美克服了高維度下的維度災難（Curse of Dimensionality）。

        Parameters:
        -----------
        df : pd.DataFrame
            輸入的資料集。
        contamination : float, default 0.05
            預期資料集中異常值所佔的比例（污染率）。

        Returns:
        --------
        pd.DataFrame
            新增 'is_outlier_iforest' 欄位（布林值）的 DataFrame。
        """
        df_res = df.copy()
        numeric_cols = df_res.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            df_res["is_outlier_iforest"] = False
            return df_res

        # 為了避免缺失值干擾演算法，這裡在模型擬合前進行簡單的填充（臨時處理，不影響原 df 數值）
        X = df_res[numeric_cols].fillna(df_res[numeric_cols].median())

        # 建立與訓練隔離森林 (設定 random_state 確保結果可複現)
        iso_forest = IsolationForest(
            contamination=contamination, random_state=42, n_jobs=-1
        )
        # sklearn 回傳: 1 代表正常, -1 代表異常
        preds = iso_forest.fit_predict(X)

        df_res["is_outlier_iforest"] = preds == -1
        print(f"[Audit: 隔離森林] 偵測到異常值數量: {df_res['is_outlier_iforest'].sum()} 筆 (設定污染率: {contamination})")
        return df_res


# =====================================================================
# MAIN 執行進入點
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 啟動自動化資料審計與異常偵測模組")
    print("=" * 60)

    # 1. 驗證資料路徑
    if not DATA_PATH.strip() or "請在此輸入" in DATA_PATH:
        print("[Error] 偵測為預設預留字串。請將最頂端的 DATA_PATH 變數替換為您電腦中的真實資料路徑！")

    elif not os.path.exists(DATA_PATH):
        print(f"[Error] 找不到檔案，請確認路徑或檔名是否拼錯：\n路徑: '{DATA_PATH}'")

    else:
        file_extension = os.path.splitext(DATA_PATH)[1].lower()

        try:
            # 2. 泛用型檔案讀取
            if file_extension == ".csv":
                df_raw = pd.read_csv(DATA_PATH)
            elif file_extension in [".xlsx", ".xls"]:
                df_raw = pd.read_excel(DATA_PATH)
            else:
                df_raw = pd.read_csv(DATA_PATH)

            print(f"📋 成功載入目標資料集！基礎規格：")
            print(f"   - 總列數 (Rows): {df_raw.shape[0]}")
            print(f"   - 總欄位數 (Columns): {df_raw.shape[1]}")
            print("-" * 60)

            # 3. 實例化審計類別
            auditor = DataAuditor()

            # 4. 依序執行審計流程
            print("\n[Step 1] 執行重複值檢查與去重...")
            df_cleaned = auditor.detect_and_drop_duplicates(df_raw)

            print("\n[Step 2] 執行統計學 Z-Score 異常偵測 (常態分佈假設)...")
            df_z = auditor.detect_outliers_zscore(df_cleaned, threshold=3.0)

            print("\n[Step 3] 執行四分位距 IQR 穩健異常偵測 (無母數防禦)...")
            df_iqr = auditor.detect_outliers_iqr(df_cleaned, factor=1.5)

            print("\n[Step 4] 執行機器學習 Isolation Forest 隔離森林偵測...")
            df_final = auditor.detect_outliers_isolation_forest(df_cleaned, contamination=0.03)

            # 5. 彙整審計綜合報告
            print("\n" + "=" * 60)
            print("📊 資料審計與異常偵測綜合報告 (Auditing Summary)")
            print("=" * 60)
            print(f"原創資料總筆數: {df_raw.shape[0]} 條")
            print(f"剔除重複後筆數: {df_cleaned.shape[0]} 條")
            print(f"⚠️ Z-Score 法判定異常數 : {df_z['is_outlier_zscore'].sum()} 筆")
            print(f"⚠️ IQR 四分位法判定異常數: {df_iqr['is_outlier_iqr'].sum()} 筆")
            print(f"⚠️ 隔離森林演算法判定異常數: {df_final['is_outlier_iforest'].sum()} 筆")
            print("-" * 60)
            print("💡 後續計量分析建議：")
            print("   1. 檢視三種方法重疊標記的樣本，若皆判定為異常，強烈建議人工核對或予以排除。")
            print("   2. 股票價格或交易量等時間序列通常具備重尾（Heavy-tailed）特徵，請優先以 IQR 或隔離森林的結果為主。")
            print("=" * 60)

        except Exception as e:
            print(f"💥 讀取或審計執行時發生未預期錯誤: {e}")
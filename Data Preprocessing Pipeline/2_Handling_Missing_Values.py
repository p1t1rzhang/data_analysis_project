import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


class MissingValueHandler:
    """精通資料工程與機器學習前處理的缺失值處理模組。

    提供三種核心缺失值處理策略：直接丟棄法、統計值填充法、以及 KNN 演算法預測填充法。
    """

    def __init__(self):
        pass

    def handle_missing_by_deletion(
        self, df: pd.DataFrame, critical_cols: list = None
    ) -> pd.DataFrame:
        """策略一：直接丟棄法 (Deletion Method)"""
        df_cleaned = df.copy()

        # 1. 橫向刪除：自動偵測並刪除缺失值比例高於 50% 的欄位
        threshold = 0.5
        cols_to_drop = df_cleaned.columns[
            df_cleaned.isnull().mean() > threshold
        ]
        if len(cols_to_drop) > 0:
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)
            print(f"[Info] 橫向刪除（缺失 > 50%）的欄位: {list(cols_to_drop)}")

        # 2. 縱向刪除：若核心關鍵欄位有缺失，則直接刪除該列 (Row)
        if critical_cols is not None:
            existing_critical_cols = [
                col for col in critical_cols if col in df_cleaned.columns
            ]
            before_rows = df_cleaned.shape[0]
            df_cleaned = df_cleaned.dropna(subset=existing_critical_cols)
            after_rows = df_cleaned.shape[0]
            print(
                f"[Info] 縱向刪除基於關鍵欄位 {existing_critical_cols}，刪除了 {before_rows - after_rows} 列數據。"
            )

        return df_cleaned

    def handle_missing_by_statistics(
        self, df: pd.DataFrame, strategy: str = "mean"
    ) -> pd.DataFrame:
        """策略二：統計值填充法 (Statistical Imputation)"""
        if strategy not in ["mean", "median"]:
            raise ValueError("Strategy must be either 'mean' or 'median'")

        df_imputed = df.copy()
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                if strategy == "mean":
                    fill_value = df_imputed[col].mean()
                else:
                    fill_value = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(fill_value)

        return df_imputed

    def handle_missing_by_knn(
        self, df: pd.DataFrame, n_neighbors: int = 5
    ) -> pd.DataFrame:
        """策略三：演算法預測填充法 (KNN Imputation)"""
        df_imputed = df.copy()

        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df_imputed.select_dtypes(
            exclude=[np.number]
        ).columns

        if len(numeric_cols) == 0:
            return df_imputed

        imputer = KNNImputer(n_neighbors=n_neighbors)
        numeric_imputed_array = imputer.fit_transform(df_imputed[numeric_cols])

        df_numeric_imputed = pd.DataFrame(
            numeric_imputed_array,
            columns=numeric_cols,
            index=df_imputed.index,
        )

        if len(non_numeric_cols) > 0:
            df_final = pd.concat(
                [df_numeric_imputed, df_imputed[non_numeric_cols]], axis=1
            )
            df_final = df_final[df.columns]
            return df_final
        else:
            return df_numeric_imputed


# ==========================================
# 4. 實體檔案讀取與缺失值處理流程測試
# ==========================================
if __name__ == "__main__":

    # 填入資料路徑
    read_file = ""

    # 檢查路徑是否有效
    if not read_file.strip() or not os.path.exists(read_file):
        print(
            f"[Error] 找不到檔案，請確認路徑是否正確：'{read_file}'"
        )
    else:
        file_extension = os.path.splitext(read_file)[1].lower()

        try:
            # 讀取檔案
            if file_extension == ".csv":
                df_raw = pd.read_csv(read_file)
            elif file_extension in [".xlsx", ".xls"]:
                df_raw = pd.read_excel(read_file)
            else:
                df_raw = pd.read_csv(read_file)

            print("=== [1. 原始資料集成功載入] ===")
            print(f"資料維度 (Rows, Cols): {df_raw.shape}")
            print(df_raw.head())
            print("\n" + "=" * 50 + "\n")

            # 實例化處理器
            handler = MissingValueHandler()

            # --- 測試策略一：直接丟棄法 ---
            critical_features = ["Date"]
            df_step1 = handler.handle_missing_by_deletion(
                df_raw, critical_cols=critical_features
            )
            print("\n=== [2. 執行策略一：直接丟棄法後] ===")
            print(f"剩餘維度: {df_step1.shape}")
            print("\n" + "=" * 50 + "\n")

            # --- 測試策略二：統計值填充法 (以中位數填充) ---
            df_step2 = handler.handle_missing_by_statistics(
                df_step1, strategy="median"
            )
            print("=== [3. 執行策略二：統計值填充法 (Median) 後] ===")
            print(f"目前是否有殘留缺失值: {df_step2.isnull().sum().sum()}")
            print("\n" + "=" * 50 + "\n")

            # --- 測試策略三：演算法預測填充法 (KNN) ---
            # 備註：若策略二已經將值填滿，策略三在傳入 df_step1 (仍有缺失的狀態) 時才能看出效果
            df_step3 = handler.handle_missing_by_knn(df_step1, n_neighbors=5)
            print("=== [4. 執行策略三：KNN 演算法預測填充後] ===")
            print(f"目前是否有殘留缺失值: {df_step3.isnull().sum().sum()}")

        except Exception as e:
            print(f"[Error] 執行過程中發生錯誤: {e}")
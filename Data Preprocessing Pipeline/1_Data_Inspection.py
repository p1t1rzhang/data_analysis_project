import os
import pandas as pd

# ==========================================
# 1. 請在此處填入您的資料集檔案路徑 (支援 .csv 或 .xlsx/.xls)
# ==========================================
read_file = ""
# 檢查使用者是否已填寫路徑
if not read_file.strip():
    print(
        "[Error] 請先在 read_file 變數中填入您的資料檔案路徑！"
    )
else:
    # 2. 自動判斷副檔名並讀取資料 (Simplicity First: 不做過度複雜的封裝)
    file_extension = os.path.splitext(read_file)[1].lower()

    try:
        if file_extension == ".csv":
            df = pd.read_csv(read_file)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(read_file)
        else:
            # 若為其他格式，預設嘗試用 read_csv 讀取
            df = pd.read_csv(read_file)

        # ==========================================
        # 3. 執行資料健康檢查 (Data Inspection)
        # ==========================================
        print("=== [內部觀測資訊] ===")
        print("\n--- 1. df.info() ---")
        df.info()

        print("\n--- 2. df.describe() ---")
        # 包含所有型態的敘述統計
        df_summary = df.describe(include="all")
        print(df_summary)

        print("\n--- 3. df.head() ---")
        print(df.head())

        # ==========================================
        # 4. 自動化彙整報告 (Output Format)
        # ==========================================
        print("\n" + "=" * 40)
        print("📊 資料集概觀")
        print("=" * 40)
        print(f"- 總列數 (Rows): {df.shape[0]}")
        print(f"- 總欄位數 (Columns): {df.shape[1]}")
        print(
            f"- 佔用記憶體大小: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        print("\n" + "=" * 40)
        print("🔍 潛在問題清單（關鍵）")
        print("=" * 40)

        # 欄位缺失值統計
        missing_count = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame(
            {"Missing Count": missing_count, "Percentage (%)": missing_pct}
        )
        missing_fields = missing_df[missing_df["Missing Count"] > 0]

        if not missing_fields.empty:
            print("[缺失值欄位與比例]")
            print(missing_fields.to_string())
        else:
            print("- 恭喜！此資料集無缺失值。")

        print("\n[資料型態檢視]")
        print(df.dtypes)
        print(
            "👉 請檢查：是否有日期被誤判為 object(字串)，或類別(Category)被誤判為 int/float？"
        )

        print("\n[極端值與合理範圍提示]")
        print(
            "👉 請檢視上方 df.describe() 的 min 與 max 值，確認是否有異常值（例如：年齡為負數或 999）。"
        )

        print("\n" + "=" * 40)
        print("💡 後續處理建議")
        print("=" * 40)
        print("1. 若有缺失值：需決定填補（Imputation）或刪除（Drop）。")
        print(
            "2. 若型態不符：建議使用 pd.to_datetime() 或 astype('category') 進行轉換。"
        )
        print("3. 若有極端值：需透過邏輯過濾（Filtering）或偏態轉換處理。")

    except Exception as e:
        print(f"[Error] 讀取檔案或分析時發生錯誤: {e}")
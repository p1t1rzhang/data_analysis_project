import os
import pandas as pd

# =====================================================================
# ⚠️ CRITICAL REQUIREMENT: DATA PATH PLACEHOLDER
# 請在此處輸入您的資料檔案路徑 (支援 .csv 或 .xlsx / .xls)
# =====================================================================
DATA_PATH = ""
    """資料特徵工程與資料轉型的模組。

    目標：把所有資料的規格調整到同一條水平線上。
    提供：日期標準化、類別資料編碼、文字雜訊清理等功能。
    """

    def __init__(self):
        pass

    def transform_date_format(
        self, df: pd.DataFrame, date_cols: list
    ) -> pd.DataFrame:
        """策略一：日期標準化 (Date Format Standardization)

        【操作的轉型原理與直覺本質】
        電腦其實看不懂「2026/06/14」、「14-Jun-26」或是「2026.06.14」這三種寫法是指同一天。
        在電腦眼中，它們只是三串毫無關聯的文字。這就是為什麼我們需要透過 pd.to_datetime()，
        強制把它們「翻譯」成標準的系統時間戳記 (System DateTime)。

        【統計/計量考量】
        在時間序列 (Time Series) 分析或計量經濟學的 Panel Data 模型中，時間是絕對的軸心。
        標準化為 DateTime 型態後，資料才真正獲得了「時間的靈魂」。我們才能夠：
        1. 計算時間差 (Time Delta)，例如：算出購買與退貨間隔了幾天。
        2. 進行時間維度的聚合，例如：自動將日資料按月或按季加總 (Resampling)。
        3. 提取特徵，例如：把時間拆解成「星期幾」、「月份」、「是否為週末」。

        Parameters:
        -----------
        df : pd.DataFrame
            輸入的資料集。
        date_cols : list
            需要轉換為 DateTime 格式的欄位名稱清單。

        Returns:
        --------
        pd.DataFrame
            日期已標準化轉換的資料集。
        """
        df_transformed = df.copy()

        for col in date_cols:
            if col in df_transformed.columns:
                try:
                    # 使用 errors='coerce' 遇到無法解析的日期會轉為 NaT，避免程式崩潰
                    df_transformed[col] = pd.to_datetime(
                        df_transformed[col], errors="coerce"
                    )
                    print(f"[Transform: 日期] 成功將欄位 '{col}' 轉換為 DateTime 型態。")
                except Exception as e:
                    print(f"[Warning: 日期] 轉換欄位 '{col}' 時發生錯誤: {e}")
            else:
                print(f"[Warning: 日期] 找不到指定欄位: '{col}'")

        return df_transformed

    def transform_categorical_encoding(
        self,
        df: pd.DataFrame,
        one_hot_cols: list = None,
        label_mapping: dict = None,
    ) -> pd.DataFrame:
        """策略二：類別資料編碼 (Categorical Data Encoding)

        【操作的轉型原理與直覺本質】
        機器學習與計量回歸模型本質上是一台巨大的數學計算機，它們只懂數字（加減乘除），看不懂文字。
        如果資料裡有「蘋果」、「香蕉」或是「低」、「高」，我們必須把它們翻譯成數字密碼。

        【統計/計量考量】
        1. Label Encoding (標籤編碼)：適用於「有順序、有大小關係」的類別。
           例如：會員等級（銅、銀、金），我們可以對應給定 0, 1, 2。
           模型在計算時會理解「金 (2) > 銀 (1) > 銅 (0)」，這符合現實的商業邏輯。
        2. One-Hot Encoding (獨熱編碼/建立虛擬變數)：適用於「無順序關係」的類別。
           例如：狗、貓、鳥。如果我們硬把它們變成 1, 2, 3，模型會誤以為「鳥(3) = 狗(1) + 貓(2)」，
           或認為鳥的價值比狗大，這在數學與邏輯上是極度荒謬的。
           因此我們利用 Dummy Variables（虛擬變數），把它們展開成三個獨立的欄位 (is_狗, is_貓, is_鳥)，
           只有是該類別的會標記為 1，其餘為 0。這樣一來，各個類別就在平等的水平線上了。

        Parameters:
        -----------
        df : pd.DataFrame
            輸入的資料集。
        one_hot_cols : list, optional
            需要進行 One-Hot Encoding（無序）的欄位清單。
        label_mapping : dict, optional
            需要進行 Label Encoding（有序）的欄位與其映射規則，格式如：
            {'Size': {'Small': 0, 'Medium': 1, 'Large': 2}}

        Returns:
        --------
        pd.DataFrame
            類別已完成數字編碼的資料集。
        """
        df_transformed = df.copy()

        # 1. 處理 Label Encoding (有順序)
        if label_mapping:
            for col, mapping in label_mapping.items():
                if col in df_transformed.columns:
                    # 使用 map 來將字典對應的值賦予上去
                    df_transformed[col] = df_transformed[col].map(mapping)
                    print(f"[Transform: 編碼] 成功對欄位 '{col}' 進行 Label Encoding。")

        # 2. 處理 One-Hot Encoding (無順序)
        if one_hot_cols:
            existing_cols = [col for col in one_hot_cols if col in df_transformed.columns]
            if existing_cols:
                # get_dummies 會自動把指定欄位展開成多個 0/1 欄位
                df_transformed = pd.get_dummies(
                    df_transformed, columns=existing_cols, dtype=int
                )
                print(f"[Transform: 編碼] 成功對欄位 {existing_cols} 進行 One-Hot Encoding。")

        return df_transformed

    def clean_text_noise(
        self, df: pd.DataFrame, text_cols: list
    ) -> pd.DataFrame:
        """策略三：文字雜訊清理 (Text Noise Cleaning)

        【操作的轉型原理與直覺本質】
        在人類看來，「$100」、「100 」、「  100」都是指同一筆金額。
        但在電腦的世界裡，字串是按字元排列的。只要多了一個空格或特殊符號，電腦就會把它當作純文字 (String)，
        拒絕把它視為可以計算的數字 (Integer/Float)。
        這就是為什麼有時候你明明看到一整排數字，卻無法算平均數或加總。

        【統計/計量考量】
        這是一個「解鎖數值本質」的過程。我們利用 Regex（正規表示式）暴力刮除文字前後的多餘空格（.strip()），
        並且把干擾性的貨幣符號（$）、千分位逗號（,）或單位標籤（%）通通剝除。
        資料清理乾淨後，我們才能將其轉換為模型渴望的純粹數值型態，真正發揮其統計價值。

        Parameters:
        -----------
        df : pd.DataFrame
            輸入的資料集。
        text_cols : list
            需要清理文字雜訊的欄位清單。

        Returns:
        --------
        pd.DataFrame
            文字雜訊已清除的資料集。
        """
        df_transformed = df.copy()

        for col in text_cols:
            if col in df_transformed.columns and df_transformed[col].dtype == "object":
                # 1. 刮除字串頭尾的隱藏空白與換行符號
                df_transformed[col] = df_transformed[col].str.strip()

                # 2. 利用正規表示式移除常見的特殊符號 (如貨幣符號、逗號等，但保留數字與小數點)
                # 正則解讀：[^0-9\.\-] 代表「非數字、非小數點、非負號」的東西，全部替換成空字串
                # 注意：如果您的欄位是純文字(姓名、地址)，不適合用此邏輯！此處邏輯偏向「解鎖被文字封印的數值」。
                # 為了安全與泛用性，這裡僅展示去除貨幣符號 $ 與千分位 , 的例子：
                df_transformed[col] = df_transformed[col].str.replace(
                    r"[\$\,]", "", regex=True
                )
                print(f"[Transform: 清理] 成功清除欄位 '{col}' 的字串雜訊 ($ 或逗號)。")
            else:
                print(f"[Warning: 清理] 欄位 '{col}' 不存在或非字串型態，略過清理。")

        return df_transformed


# =====================================================================
# MAIN 執行進入點
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 啟動自動化資料特徵工程與轉型模組")
    print("=" * 60)

    # 1. 驗證資料路徑
    if not DATA_PATH.strip() or "請在此輸入" in DATA_PATH:
        print(
            "[Error] 偵測為預設預留字串。請將最頂端的 DATA_PATH 變數替換為您電腦中的真實資料路徑！"
        )

    elif not os.path.exists(DATA_PATH):
        # 為了能展示這個模組的威力，如果在您的電腦找不到檔案，我們動態生成一個 Dummy DataFrame 來測試
        print(f"[Info] 找不到實體檔案 '{DATA_PATH}'。")
        print("[Info] 系統自動建立測試專用 Dummy 資料集展示轉型效果...\n")
        
        dummy_data = {
            "Date_Str": ["2026/06/14", "15-Jun-26", "2026.06.16", "2026-06-17"],
            "Risk_Level": ["Low", "Medium", "High", "Low"],  # 有序類別
            "Sector": ["Tech", "Finance", "Tech", "Energy"], # 無序類別
            "Revenue": ["$1,000.50", " 2,500.00 ", "$3,000", "4,200.75 "], # 充滿雜訊的數值
        }
        df_raw = pd.DataFrame(dummy_data)

    else:
        file_extension = os.path.splitext(DATA_PATH)[1].lower()
        try:
            if file_extension == ".csv":
                df_raw = pd.read_csv(DATA_PATH)
            elif file_extension in [".xlsx", ".xls"]:
                df_raw = pd.read_excel(DATA_PATH)
            else:
                df_raw = pd.read_csv(DATA_PATH)
            print(f"📋 成功載入目標資料集！")
        except Exception as e:
            print(f"💥 讀取檔案失敗: {e}")
            exit()

    print("\n--- 原始資料 (Raw Data) ---")
    print(df_raw)
    print("\n[原始資料型態]")
    print(df_raw.dtypes)
    print("-" * 60)

    # 3. 實例化轉型類別
    transformer = DataTransformer()
    df_processed = df_raw.copy()

    # 4. 依序執行特徵轉型流程
    print("\n[Step 1] 執行日期標準化...")
    # 💡 夥伴注意：請將 "Date_Str" 替換成您股票資料中的日期欄位（例如 "Date"）
    df_processed = transformer.transform_date_format(
        df_processed, date_cols=["Date_Str"]
    )

    print("\n[Step 2] 執行類別資料編碼...")
    # 定義 Label Mapping (有大小順序的對應關係)
    mapping_dict = {"Risk_Level": {"Low": 0, "Medium": 1, "High": 2}}
    # 執行編碼
    df_processed = transformer.transform_categorical_encoding(
        df_processed,
        one_hot_cols=["Sector"],  # 無順序，展開為 Dummy Variables
        label_mapping=mapping_dict # 有順序，映射為 0, 1, 2
    )

    print("\n[Step 3] 執行文字雜訊清理...")
    # 💡 夥伴注意：這會移除文字中的 $ 與 ,
    df_processed = transformer.clean_text_noise(
        df_processed, text_cols=["Revenue"]
    )
    
    # 補充：清理完文字雜訊後，通常會緊接著將其轉型為 float 數字型態，才能進模型
    if "Revenue" in df_processed.columns:
        try:
             df_processed["Revenue"] = df_processed["Revenue"].astype(float)
             print("[Transform: 終極解鎖] 成功將 'Revenue' 從乾淨文字轉化為純數值 (Float)。")
        except Exception as e:
             pass

    # 5. 彙整轉型對比報告
    print("\n" + "=" * 60)
    print("✨ 資料特徵工程與轉型完成 (Transformation Complete)")
    print("=" * 60)
    print("\n--- 轉型後資料 (Transformed Data) ---")
    print(df_processed)
    print("\n[轉型後資料型態]")
    print(df_processed.dtypes)
    print("=" * 60)
    
    # 6. (可選) 匯出資料
    OUTPUT_PATH = ""
    df_processed.to_csv(OUTPUT_PATH, index=False)
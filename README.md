# 📊 Data Analysis Project |


---

## 📂 專案目錄與研究主題


### 🧪 [Data Preprocessing Pipeline](./Data%20Preprocessing%20Pipeline/)
* **專案簡介**：從平常專案的直覺思維出發，建立的五階段資料前處理與清洗工作流。
* **核心技術**：
  1. **資料健康檢查 (Data Inspection)**：利用 `df.info()`, `df.describe()` 剖析資料體質。
  2. **缺失值處理 (Handling Missing Values)**：結合中位數防禦（針對偏態資料）與 KNN 鄰居物以類聚填充。
  3. **異常與重複偵測 (Outlier Detection)**：以 Z-Score (常態假設) 與 IQR (無母數防禦) 雙重卡出數據防線，並用孤立森林（Isolation Forest）切分極端值。
  4. **資料型態與格式統一 (Data Transformation)**：DateTime 標準化與虛擬變數（One-Hot/Label Encoding）計量轉型。
  5. **特徵縮放 (Feature Scaling)**：使用 StandardScaler 與 MinMaxScaler 確保特徵在模型中公平競爭。

---

## 🛠️ 開發環境與核心工具

主要基於以下環境進行開發：
* **作業系統**：macOS (Terminal / iTerm2 工作流)
* **核心語言**：Python 3.x
---


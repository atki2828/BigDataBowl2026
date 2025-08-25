# 🏈 Big Data Bowl 2026 

This repository contains analysis, modeling, and visualization for the **NFL Big Data Bowl 2026** project.  

We use **[uv](https://github.com/astral-sh/uv)** for Python dependency management to keep the environment fast and consistent across all platforms.

---

## ✅ Project Setup Instructions

### **1. Clone the Repository**
```bash
git clone https://github.com/YOUR-USERNAME/BigDataBowl2026.git
cd BigDataBowl2026
```

### **2. Install uv (Dependency Manager)**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```
### **3. Setup Environment**
```bash
uv sync
```
### **4. Activate Virtual Environment**
```bash
source .venv/bin/activate
```
### **5. Activate Virtual Environment**
```bash
uv run python Hello.py
```



## 🖥️ Windows Setup Instructions

### **1. Clone the Repository**
```powershell
git clone https://github.com/YOUR-USERNAME/BigDataBowl2026.git
cd BigDataBowl2026
```   

### **2. Install uv (Dependency Manager)**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### **3. Setup Environment**
```powershell
uv sync
```
### **4. Activate Virtual Environment**
```powershell
.venv\Scripts\Activate.ps1
```

### **5. Run a Test Script**
```powershell
uv run python Hello.
```


## 🔐 How to Get Your Databricks Credentials

### **1. Generate a Personal Access Token**
1. After logging into the workspace, click your **user icon → User Settings**.  
2. Go to the **Access tokens** tab.  
3. Click **Generate new token** and copy it.  
4. Save this somewhere safe — you’ll need it for connecting with the REST API or Python/SQL connectors.  

---

### **2. Find Your Hostname and HTTP Path**
1. Go to your **Databricks workspace**.  
2. In the left sidebar, click **SQL Warehouses** (or **Clusters** if you’re using classic clusters).  
3. Choose the warehouse/cluster you’ll connect to.  
4. Click the **Connection details** tab.  
   - **DATABRICKS_SERVER_HOSTNAME** will look like:  
     ```
     adb-123456789012345.7.azuredatabricks.net
     ```  
   - **DATABRICKS_HTTP_PATH** will look like:  
     ```
     /sql/1.0/warehouses/abcdef1234567890
     ```  

---

### **3. Set Environment Variables**

#### On **Linux / macOS (bash/zsh)**
```bash
export DATABRICKS_SERVER_HOSTNAME="adb-123456789012345.7.azuredatabricks.net"
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/abcdef1234567890"
export DATABRICKS_TOKEN="paste-your-token-here"
```

#### On **Windows (PowerShell)**
```powershell
setx DATABRICKS_SERVER_HOSTNAME "adb-123456789012345.7.azuredatabricks.net"
setx DATABRICKS_HTTP_PATH "/sql/1.0/warehouses/abcdef1234567890"
setx DATABRICKS_TOKEN "paste-your-token-here"
```

### To test working run 
```python
uv run .\tests\test_dbx.py
```
**This should return a polars df of shape (4,9)**
**## Multi-Label Email Classification (Chained Multi-Output Architecture)**
The implementations in this repository primarily focus on the **Chained Multi-Output Architecture** for multi-label email classification. In this approach, the machine learning model is used to predict the dependent variable sequentially, where each prediction depends on the previous one:
- First: Type 2  
- Second: Type 2 + Type 3  
- Third: Type 2 + Type 3 + Type 4
This design captures the dependencies between labels and ensures that the predictions follow a structured chained structure.
---
**# Environment Setup**

Follow the steps below to set up the environment using Conda:

```bash
conda create -n engineering python=3.10 -y
conda activate engineering
---
**# Install Dependencies**
Install the required Python libraries:
```bash
conda install pandas
conda install scikit-learn
---
**# Navigate to Project Directory**
```bash
cd /path/to/your/project

> Please replace `/path/to/your/project` with the path to your project on your computer, for example:
>
> - **macOS:** `/Users/yourname/Desktop/your_project`
> - **Windows:** `C:\Users\yourname\Desktop\your_project`
---
**# Run the Project**
Execute the main script:
```bash
python main.py

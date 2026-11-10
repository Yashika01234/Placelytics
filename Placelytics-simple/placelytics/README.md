<div align="center">

# 📊 Placelytics ✨
### A simple campus placement analysis project

![Python](https://img.shields.io/badge/Python-Pandas-6C8CFF?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-F472B6?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

> 🌸 All data is **synthetic**, generated for a fictional college. It is **not** real placement data.

## 🎯 Questions this project answers
1. What is the overall placement rate, and how does it differ by **branch**?
2. What do package numbers look like (median vs average vs highest)?
3. Do placed students have a higher **CGPA**?
4. Do **internships** help, even when comparing students with similar CGPA?
5. Which companies hired the most?

## 🛠️ What I did
- Generated a 800-row synthetic dataset (`generate_data.py`) with some missing values and duplicates
- **Cleaned** it: removed duplicates, filled missing CGPA (branch median) and missing package (company median)
- Analysed it with **Pandas** (`analysis.py`) and made 5 **Matplotlib** charts
- Built a small interactive **Streamlit** dashboard (`app.py`) with filters and CSV download

## 🚀 Run it
```bash
python -m venv .venv
.venv\Scripts\activate           # Windows   (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt

python generate_data.py          # creates data/placements.csv
python analysis.py               # prints results, saves charts/ images
python -m streamlit run app.py   # dashboard at http://localhost:8501
```
💡 In VS Code, open `analysis.py` and click **Run Cell** above each `# %%` block to use it like a notebook.

## 📈 Sample findings (synthetic data)
- Overall placement rate is **66.8%**; median package **7.97 LPA**, highest **15.6 LPA**
- **IT** has the highest placement rate (80.7%) and **ME** the lowest (44.5%)
- Placed students average a CGPA of 7.84 vs 7.17 for non-placed
- Students with an internship were placed **80%** of the time vs **54%** without (the gap holds within each CGPA band too)
- Mean and median packages are close (about 8 LPA), so there are no big outliers in this dataset

## 📂 Files
```
Placelytics/
├── generate_data.py   # synthetic data
├── analysis.py        # cleaning + analysis + charts
├── app.py             # streamlit dashboard
├── data/placements.csv
└── charts/            # saved charts
```

## ⚠️ Limitations
Synthetic data shows the relationships I built in, so it demonstrates the *method*, not real-world results. The internship result is an association, not proof of cause.

## 💌 Author
**Yashika** · [GitHub](https://github.com/Yashika01234) · [LinkedIn](https://www.linkedin.com/in/yashika-nanda-936414321)

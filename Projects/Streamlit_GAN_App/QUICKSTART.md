# Quick Start Guide - CTGAN with CSV Input

## 🚀 Get Started in 30 Seconds

### Step 1: Prepare Your Data
Place your data in a CSV file. Example format:

```csv
Age,Sex,Income,Category,Outcome
45,M,50000,A,Good
52,F,62000,B,Bad
38,M,45000,A,Good
```

### Step 2: Run Training
```bash
python train_from_csv.py your_data.csv
```

### Step 3: Review Results
The script will:
1. Show a data quality report
2. Ask if it looks correct (type `yes` or `no`)
3. Train the model
4. Generate synthetic data
5. Save everything to `training_output/`

---

## 📊 What Happens Step-by-Step

### 1. Data Loading & Analysis
```
Loading CSV: your_data.csv...

======================================================================
INITIAL DATA REPORT
======================================================================
File: your_data.csv
Rows: 500
Columns: 6
Total missing values: 12
```

The system reads your CSV and reports:
- Total rows and columns
- How many missing values (and where)
- Column names

### 2. Data Quality Report
```
======================================================================
DATA QUALITY REPORT
======================================================================
Rows before cleaning: 500
Rows dropped (due to NAs): 12
Rows after cleaning: 488
Retention rate: 97.6%

Column Classification:
  Continuous columns (2): Age, Income
  Categorical columns (2): Sex, Category
  Binary columns (2): Outcome
```

The system shows:
- How many rows will be used
- What percentage of data is retained
- How columns are classified

### 3. User Confirmation
```
Does this data look correct? (yes/no): yes
✓ Proceeding with training...
```

You can review the data quality and decide whether to proceed.

### 4. Model Training
```
Training CTGAN...
Epoch [1/10] Iter 50 d_loss=-120.4352 g_loss=-45.2134
Epoch [1/10] Iter 100 d_loss=-125.6341 g_loss=-48.9234
...
✓ Training complete. Total iterations: 500
```

The model trains on your cleaned data.

### 5. Evaluation & Visualization
```
Computing evaluation metrics...

============================================================
CTGAN Evaluation Metrics
============================================================
RMSE: 12.345678
MSE:  152.415000
MAE:  8.123456

Column-wise Statistics:
Age:
  Real Mean: 45.2300, Synth Mean: 45.1234, Diff: 0.1066
  Real Std:  15.6200, Synth Std:  15.4321, Diff: 0.1879
```

The system computes quality metrics and creates visualizations.

### 6. Output Files
```
============================================================
✓ All outputs saved to: training_output/
  - losses.png: Training loss curves
  - pca.png: PCA visualization of real vs synthetic
  - distributions_continuous.png: Distribution comparisons
  - distributions_categorical.png: Category proportions
  - ctgan_model.pth: Trained model (can be loaded later)
  - synthetic_data.csv: Generated synthetic dataset
  - real_data_cleaned.csv: Cleaned training data
  - training_report.txt: Full summary
============================================================
```

All results are saved for you to review.

---

## 🎯 Common Use Cases

### Use Case 1: Simple Training
```bash
python train_from_csv.py customer_data.csv
```

### Use Case 2: More Training Epochs
```bash
python train_from_csv.py customer_data.csv 20
```

### Use Case 3: Programmatic Usage
```python
from data_loader import DataLoader
from CTGAN_dec_adjustable import CustomCTGAN

# Load data
loader = DataLoader('my_data.csv')
df, report = loader.load_and_analyze(interactive=False)

# Get column types
continuous, categorical, binary = loader.get_column_types()

# Train model
model = CustomCTGAN(
    continuous_cols=continuous,
    categorical_cols=categorical,
    binary_cols=binary,
    epochs=15
)
model.fit(df)

# Generate synthetic data
synthetic = model.sample(1000)
synthetic.to_csv('output.csv', index=False)
```

---

## 📋 Data Format Requirements

### ✓ What Works
- Standard CSV files
- Any numeric types (int, float)
- Text/string values for categories
- Column headers in first row

### ✗ What Doesn't Work
- Excel files (.xlsx) - convert to CSV first
- Rows with missing values - they will be dropped
- Empty columns - remove them first

### ⚠️ Best Practices
- Use meaningful column names
- Remove sensitive/identifying information
- Ensure data is in the right units (dates as numbers, etc.)
- Check for outliers or data quality issues

---

## 🔍 Column Type Auto-Detection

The system automatically classifies columns:

| Type | Detection Rule | Example |
|------|---|---|
| **Continuous** | Numeric with 11+ unique values | Age, Income, Score |
| **Categorical** | Text or numeric with 3-10 unique values | City, Ethnicity, Grade |
| **Binary** | Exactly 2 unique values (any type) | Sex (M/F), Outcome (Y/N) |

You can override this in the code if needed.

---

## 🛠️ Customization

### Adjust Training Parameters

Edit `train_from_csv.py` line with `CustomCTGAN()`:

```python
model = CustomCTGAN(
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols,
    binary_cols=binary_cols,
    condition_col=condition_col,
    noise_dim=64,                    # Size of random noise
    generator_dim=(256, 256),        # Hidden layer sizes
    discriminator_dim=(256, 256),    # Hidden layer sizes
    batch_size=128,                  # Batch size
    epochs=10,                       # Training epochs
    lr_g=2e-4,                       # Learning rate (generator)
    lr_d=2e-4,                       # Learning rate (discriminator)
    n_critic=5,                      # D updates per G update
    wgan_gp=True,                    # Use gradient penalty
    dropout=0.1,                     # Dropout rate
    verbose=True                     # Show progress
)
```

---

## ❓ FAQ

### Q: What if my CSV has missing values?
**A:** The system drops rows with any missing values and reports how many. You'll see the retention rate in the data quality report.

### Q: What's a "condition column"?
**A:** It's used for conditional generation - you can generate synthetic data with specific values for that column.

### Q: How long does training take?
**A:** ~1-5 seconds per epoch on CPU. With 10 epochs and small data: ~30-60 seconds total.

### Q: Can I use a very small dataset?
**A:** Yes, but with <50 rows, results may not be great. Aim for 100+ rows for better quality.

### Q: What's in the synthetic_data.csv?
**A:** Completely artificial data that matches the statistical properties of your real data. It's safe to share.

### Q: Can I load a saved model?
**A:** Yes! Use:
```python
model = CustomCTGAN(...)
model.load('training_output/ctgan_model.pth')
synthetic = model.sample(1000)
```

---

## 📚 Example Files

Two examples are provided:

1. **sample_data.csv** (25 rows, clean)
   - No missing values
   - Good for quick testing

2. **sample_data_with_nas.csv** (25 rows, 7 missing values)
   - Shows the filtering in action
   - 28% of rows dropped
   - 72% retention rate

---

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_train_csv.py
```

This trains a small model on sample data and reports metrics.

---

## 📞 Troubleshooting

| Problem | Solution |
|---------|----------|
| "CSV file not found" | Check file path, use absolute path if needed |
| Prompt doesn't appear | Make sure terminal is interactive |
| Training is slow | Use fewer epochs: `python train_from_csv.py data.csv 3` |
| Memory error | Use smaller batch size, or reduce model size |
| Low quality synthetic data | Use more epochs or larger training data |

---

## 📖 Full Documentation

See `CSV_README.md` for comprehensive documentation.

See `IMPLEMENTATION_SUMMARY.md` for technical details.

---

## ✨ You're Ready!

```bash
# Create your CSV file (e.g., mydata.csv)
# Then run:
python train_from_csv.py mydata.csv

# Answer "yes" when prompted
# Check training_output/ for results!
```

Enjoy generating synthetic data! 🎉

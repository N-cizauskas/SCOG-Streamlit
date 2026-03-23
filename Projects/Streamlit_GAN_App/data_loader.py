"""
Flexible data loader for CTGAN with auto-detection of column types and data quality reporting.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional


class DataLoader:
    """Load CSV data with auto-detection of column types and interactive data quality confirmation."""

    def __init__(self, csv_path: str, min_unique_threshold: int = 10):
        """
        Initialize DataLoader.
        
        Args:
            csv_path: Path to CSV file
            min_unique_threshold: Threshold to distinguish categorical from continuous.
                                 If unique values <= threshold, treat as categorical.
        """
        self.csv_path = csv_path
        self.min_unique_threshold = min_unique_threshold
        self.original_df = None
        self.cleaned_df = None
        self.continuous_cols = []
        self.categorical_cols = []
        self.binary_cols = []

    def load_and_analyze(self, interactive: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Load CSV, detect types, drop NAs, and optionally prompt user for confirmation.
        
        Returns:
            Tuple of (cleaned dataframe, report dict)
        """
        # load CSV
        print(f"Loading CSV: {self.csv_path}...")
        self.original_df = pd.read_csv(self.csv_path)
        
        # generate initial report
        report = self._generate_initial_report()
        self._print_initial_report(report)
        
        # auto-detect column types
        self._detect_column_types()
        
        # clean data (drop NAs)
        self.cleaned_df = self.original_df.dropna()
        
        # generate final report
        report['rows_dropped'] = len(self.original_df) - len(self.cleaned_df)
        report['rows_final'] = len(self.cleaned_df)
        report['continuous_cols'] = self.continuous_cols
        report['categorical_cols'] = self.categorical_cols
        report['binary_cols'] = self.binary_cols
        report['all_feature_cols'] = self.continuous_cols + self.categorical_cols + self.binary_cols
        
        self._print_data_quality_report(report)
        
        # ask user for confirmation
        if interactive:
            if not self._confirm_with_user():
                raise RuntimeError("User rejected data quality report. Aborting.")
        
        return self.cleaned_df, report

    def _generate_initial_report(self) -> Dict:
        """Generate initial report before cleaning."""
        return {
            'csv_path': self.csv_path,
            'rows_initial': len(self.original_df),
            'columns': list(self.original_df.columns),
            'n_columns': len(self.original_df.columns),
            'missing_values': self.original_df.isnull().sum().to_dict(),
            'total_missing': self.original_df.isnull().sum().sum()
        }

    def _detect_column_types(self):
        """Auto-detect column types: continuous, categorical, or binary."""
        self.continuous_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        
        for col in self.original_df.columns:
            col_data = self.original_df[col].dropna()
            
            # skip if all NAs
            if len(col_data) == 0:
                continue
            
            # check if numeric
            if pd.api.types.is_numeric_dtype(col_data):
                n_unique = col_data.nunique()
                
                # binary: exactly 2 unique values
                if n_unique == 2:
                    self.binary_cols.append(col)
                # continuous: many unique values
                else:
                    self.continuous_cols.append(col)
            else:
                # non-numeric: treat as categorical
                n_unique = col_data.nunique()
                
                # binary: exactly 2 unique values
                if n_unique == 2:
                    self.binary_cols.append(col)
                # categorical: multiple categories
                else:
                    self.categorical_cols.append(col)

    def _print_initial_report(self, report: Dict):
        """Print initial data report."""
        print("\n" + "=" * 70)
        print("INITIAL DATA REPORT")
        print("=" * 70)
        print(f"File: {report['csv_path']}")
        print(f"Rows: {report['rows_initial']}")
        print(f"Columns: {report['n_columns']}")
        print(f"Total missing values: {report['total_missing']}")
        
        if report['total_missing'] > 0:
            print("\nMissing values by column:")
            for col, count in report['missing_values'].items():
                if count > 0:
                    print(f"  {col}: {count} ({100*count/report['rows_initial']:.1f}%)")
        
        print(f"\nColumn names: {', '.join(report['columns'])}")

    def _print_data_quality_report(self, report: Dict):
        """Print detailed data quality report."""
        print("\n" + "=" * 70)
        print("DATA QUALITY REPORT")
        print("=" * 70)
        print(f"Rows before cleaning: {report['rows_initial']}")
        print(f"Rows dropped (due to NAs): {report['rows_dropped']}")
        print(f"Rows after cleaning: {report['rows_final']}")
        if report['rows_initial'] > 0:
            print(f"Retention rate: {100 * report['rows_final'] / report['rows_initial']:.1f}%")
        
        print(f"\nColumn Classification:")
        print(f"  Continuous columns ({len(report['continuous_cols'])}): {', '.join(report['continuous_cols']) if report['continuous_cols'] else 'None'}")
        print(f"  Categorical columns ({len(report['categorical_cols'])}): {', '.join(report['categorical_cols']) if report['categorical_cols'] else 'None'}")
        print(f"  Binary columns ({len(report['binary_cols'])}): {', '.join(report['binary_cols']) if report['binary_cols'] else 'None'}")
        
        total_features = len(report['all_feature_cols'])
        print(f"\nTotal feature columns: {total_features}")
        print("=" * 70)

    def _confirm_with_user(self) -> bool:
        """Ask user to confirm data quality looks correct."""
        while True:
            user_input = input("\nDoes this data look correct? (yes/no): ").strip().lower()
            if user_input in ['yes', 'y']:
                print(">> Proceeding with training...\n")
                return True
            elif user_input in ['no', 'n']:
                print(">> User rejected data. Aborting.\n")
                return False
            else:
                print("Please enter 'yes' or 'no'.")

    def get_column_types(self) -> Tuple[List[str], List[str], List[str]]:
        """Return detected column types."""
        return self.continuous_cols, self.categorical_cols, self.binary_cols

    def get_data(self) -> pd.DataFrame:
        """Return cleaned dataframe."""
        if self.cleaned_df is None:
            raise RuntimeError("Data not loaded yet. Call load_and_analyze() first.")
        return self.cleaned_df

    @staticmethod
    def explain_condition_column():
        """Display explanation of condition columns."""
        explanation = """
================================================================================
WHAT IS A CONDITION COLUMN?
================================================================================

A condition column is used for CONDITIONAL GENERATION - it allows you to 
generate synthetic data with specific values for that column.

EXAMPLE:
If you have a 'Treatment' column with values ['A', 'B'], you can generate:
  • 1000 rows with Treatment = 'A'
  • 1000 rows with Treatment = 'B'
  • Or a mix of both

WHY USE IT?
  [+] Generate data for specific groups/categories separately
  [+] Ensure balanced synthetic data by condition
  [+] Study how the model learns different groups
  [+] Test model behavior across different conditions

BEST PRACTICES:
  [+] Use a column with 2-5 unique values (e.g., binary or categorical)
  [+] Avoid columns with too many categories (>10)
  [+] Avoid continuous columns (use categorical/binary instead)
  [+] Can skip this if you don't need conditional generation

================================================================================
"""
        print(explanation)

    @staticmethod
    def prompt_for_condition_column(continuous_cols: List[str], categorical_cols: List[str], 
                                    binary_cols: List[str]) -> Optional[str]:
        """Interactively prompt user to select a condition column."""
        all_cols = continuous_cols + categorical_cols + binary_cols
        
        if not all_cols:
            print("No columns available for condition selection.")
            return None
        
        # explain condition columns
        DataLoader.explain_condition_column()
        
        print("\nAvailable columns for condition:")
        print("-" * 80)
        
        # organize by type
        if binary_cols:
            print(f"\nBinary columns (BEST CHOICE - exactly 2 values):")
            for i, col in enumerate(binary_cols, 1):
                print(f"  {i}. {col}")
            choice_start = 1
            choice_binary = list(enumerate(binary_cols, 1))
        else:
            choice_start = 1
            choice_binary = []
        
        if categorical_cols:
            print(f"\nCategorical columns (GOOD - 3-10 values):")
            start = len(choice_binary) + 1
            for i, col in enumerate(categorical_cols, start):
                print(f"  {i}. {col}")
            choice_categorical = list(enumerate(categorical_cols, start))
        else:
            choice_categorical = []
        
        if continuous_cols:
            print(f"\nContinuous columns (NOT RECOMMENDED - too many unique values):")
            start = len(choice_binary) + len(choice_categorical) + 1
            for i, col in enumerate(continuous_cols, start):
                print(f"  {i}. {col} [WARNING: many unique values]")
            choice_continuous = list(enumerate(continuous_cols, start))
        else:
            choice_continuous = []
        
        print(f"\n  0. Skip - No condition column")
        print("-" * 80)
        
        while True:
            try:
                choice = input("\nSelect a column for conditional generation (0-skip, 1-{}, or column name): ".format(
                    len(choice_binary) + len(choice_categorical) + len(choice_continuous))).strip().lower()
                
                # skip option
                if choice == '0':
                    print(">> Skipping condition column.\n")
                    return None
                
                # numeric choice
                if choice.isdigit():
                    choice_num = int(choice)
                    
                    # find which choice this is
                    for idx, col in choice_binary:
                        if idx == choice_num:
                            print(f">> Using '{col}' as condition column (binary).\n")
                            return col
                    
                    for idx, col in choice_categorical:
                        if idx == choice_num:
                            print(f">> Using '{col}' as condition column (categorical).\n")
                            return col
                    
                    for idx, col in choice_continuous:
                        if idx == choice_num:
                            print(f">> Using '{col}' as condition column (continuous - may not work well).\n")
                            confirm = input("Are you sure? (yes/no): ").strip().lower()
                            if confirm in ['yes', 'y']:
                                return col
                            continue
                
                # column name
                if choice in all_cols:
                    print(f">> Using '{choice}' as condition column.\n")
                    return choice
                
                print(f"Invalid choice. Please select a valid option.")
            
            except KeyboardInterrupt:
                print("\n>> User cancelled. Skipping condition column.\n")
                return None

# Batch Auction AMM Simulator

## Overview
A comprehensive batch-auction AMM (Automated Market Maker) simulator implementing constant-product pools with uniform clearing price mechanism. Built for the Epic AMM DeFi project to analyze and compare batch auction execution vs traditional sequential AMM execution.

## Features
- **Batch Auction Mode**: All orders in a batch clear at a single uniform price, solved via quadratic equation
- **Sequential Mode**: Traditional AMM where each order executes one-by-one, affecting price for subsequent orders
- **MinOut Constraints**: Iterative filtering removes orders that fail minimum output requirements
- **Comparison Analysis**: Side-by-side comparison of batch vs sequential execution with MEV protection insights
- **Order Generation**: Manual entry or random order generation with configurable parameters

## Technical Stack
- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Math**: Python math module for quadratic equation solving

## Mathematical Specification

### Constant Product Invariant
```
k = X * Y
```

### Batch Auction Clearing
For orders with total effective inputs A (X→Y) and B (Y→X):

**Quadratic Equation for clearing price p:**
```
a*p² + b*p + c = 0
where:
  a = B(Y₀ + B)
  b = -((X₀ + A)(Y₀ + B) + AB - k)
  c = (X₀ + A)A
```

**Final Reserves:**
```
X₁ = X₀ + A - B*p
Y₁ = Y₀ + B - A/p
```

**Payouts:**
- ΔY = A/p (Y paid to X→Y traders)
- ΔX = B*p (X paid to Y→X traders)

### Edge Cases
- B = 0 (only X→Y orders): p = (X₀+A)*A / ((X₀+A)*Y₀ - k)
- A = 0 (only Y→X orders): p = (X₀*(Y₀+B) - k) / (B*(Y₀+B))

## Project Structure
```
/
├── app.py                 # Main Streamlit application with AMM simulator
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── attached_assets/       # Project documentation and specifications
└── replit.md              # This file
```

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Key Concepts

### Order Format
```json
{
  "side": "X_TO_Y" | "Y_TO_X",
  "amountIn": number,
  "minOut": number
}
```

### Batch vs Sequential
- **Batch Auction**: Eliminates MEV opportunities by clearing all orders at uniform price
- **Sequential**: Vulnerable to sandwich attacks due to order-dependent price impact

### MinOut Filtering
The simulator implements iterative minOut constraint enforcement:
1. Compute clearing price with all orders
2. Check which orders fail their minOut constraint
3. Remove failing orders and recompute
4. Repeat until all remaining orders can fill

## Recent Changes
- December 2024: Complete rebuild to match batch-auction AMM specification
- Implemented correct quadratic solver with invariant preservation
- Added solution selection to prefer non-degenerate clearing prices
- Enhanced metrics display with reserve changes and invariant checks
# BAMM

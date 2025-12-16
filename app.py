import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math

st.set_page_config(
    page_title="Batch Auction AMM Simulator",
    page_icon="🔄",
    layout="wide"
)

@dataclass
class Order:
    side: str
    amount_in: float
    min_out: float
    order_id: int = 0
    
    def to_dict(self):
        return {
            "order_id": self.order_id,
            "side": self.side,
            "amountIn": self.amount_in,
            "minOut": self.min_out
        }

@dataclass
class OrderFill:
    order: Order
    amount_out: float
    execution_price: float
    filled: bool
    fail_reason: str = ""

@dataclass
class BatchResult:
    clearing_price: float
    final_X: float
    final_Y: float
    delta_X: float
    delta_Y: float
    fills: List[OrderFill]
    initial_X: float
    initial_Y: float
    iterations: int = 1

class ConstantProductAMM:
    def __init__(self, reserve_x: float, reserve_y: float, fee_rate: float = 0.003):
        self.X = reserve_x
        self.Y = reserve_y
        self.fee_rate = fee_rate
        self.k = reserve_x * reserve_y
    
    def copy(self):
        return ConstantProductAMM(self.X, self.Y, self.fee_rate)
    
    def spot_price(self) -> float:
        return self.X / self.Y
    
    def swap_x_to_y(self, dx: float) -> float:
        dx_eff = dx * (1 - self.fee_rate)
        dy = (self.Y * dx_eff) / (self.X + dx_eff)
        self.X += dx_eff
        self.Y -= dy
        return dy
    
    def swap_y_to_x(self, dy: float) -> float:
        dy_eff = dy * (1 - self.fee_rate)
        dx = (self.X * dy_eff) / (self.Y + dy_eff)
        self.Y += dy_eff
        self.X -= dx
        return dx
    
    def execute_sequential(self, orders: List[Order]) -> List[OrderFill]:
        fills = []
        for order in orders:
            if order.side == "X_TO_Y":
                amount_out = self.swap_x_to_y(order.amount_in)
                exec_price = order.amount_in * (1 - self.fee_rate) / amount_out if amount_out > 0 else 0
                filled = amount_out >= order.min_out
                fail_reason = "" if filled else f"Output {amount_out:.4f} < minOut {order.min_out:.4f}"
            else:
                amount_out = self.swap_y_to_x(order.amount_in)
                exec_price = amount_out / (order.amount_in * (1 - self.fee_rate)) if order.amount_in > 0 else 0
                filled = amount_out >= order.min_out
                fail_reason = "" if filled else f"Output {amount_out:.4f} < minOut {order.min_out:.4f}"
            
            fills.append(OrderFill(
                order=order,
                amount_out=amount_out,
                execution_price=exec_price,
                filled=filled,
                fail_reason=fail_reason
            ))
        return fills
    
    def solve_batch_clearing_price(self, A: float, B: float) -> Optional[float]:
        X0, Y0 = self.X, self.Y
        k = self.k
        
        if A == 0 and B == 0:
            return X0 / Y0
        
        if B == 0:
            denom = (X0 + A) * Y0 - k
            if denom <= 0:
                return None
            p = (X0 + A) * A / denom
            return p if p > 0 else None
        
        if A == 0:
            denom = B * (Y0 + B)
            if denom <= 0:
                return None
            p = (X0 * (Y0 + B) - k) / denom
            return p if p > 0 else None
        
        a = B * (Y0 + B)
        b = -((X0 + A) * (Y0 + B) + A * B - k)
        c = (X0 + A) * A
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        sqrt_disc = math.sqrt(discriminant)
        p1 = (-b + sqrt_disc) / (2 * a)
        p2 = (-b - sqrt_disc) / (2 * a)
        
        valid_solutions = []
        for p in [p1, p2]:
            if p > 0:
                X1 = X0 + A - B * p
                Y1 = Y0 + B - A / p
                if X1 > 0 and Y1 > 0:
                    inv_check = abs(X1 * Y1 - k) / k
                    reserve_change = abs(X1 - X0) + abs(Y1 - Y0)
                    valid_solutions.append((p, inv_check, reserve_change, X1, Y1))
        
        if valid_solutions:
            non_degenerate = [s for s in valid_solutions if s[2] > 0.01]
            if non_degenerate:
                non_degenerate.sort(key=lambda x: x[1])
                return non_degenerate[0][0]
            valid_solutions.sort(key=lambda x: x[1])
            return valid_solutions[0][0]
        
        return None
    
    def execute_batch(self, orders: List[Order]) -> BatchResult:
        initial_X, initial_Y = self.X, self.Y
        active_orders = orders.copy()
        all_fills = {}
        iterations = 0
        max_iterations = 100
        final_p = initial_X / initial_Y
        
        for order in orders:
            all_fills[order.order_id] = OrderFill(
                order=order,
                amount_out=0,
                execution_price=0,
                filled=False,
                fail_reason="Not processed"
            )
        
        while iterations < max_iterations:
            iterations += 1
            
            A = sum(o.amount_in * (1 - self.fee_rate) for o in active_orders if o.side == "X_TO_Y")
            B = sum(o.amount_in * (1 - self.fee_rate) for o in active_orders if o.side == "Y_TO_X")
            
            if A == 0 and B == 0:
                for order in orders:
                    if order not in active_orders:
                        all_fills[order.order_id] = OrderFill(
                            order=order,
                            amount_out=0,
                            execution_price=0,
                            filled=False,
                            fail_reason="Removed in minOut filtering"
                        )
                
                return BatchResult(
                    clearing_price=final_p,
                    final_X=initial_X,
                    final_Y=initial_Y,
                    delta_X=0,
                    delta_Y=0,
                    fills=list(all_fills.values()),
                    initial_X=initial_X,
                    initial_Y=initial_Y,
                    iterations=iterations
                )
            
            p = self.solve_batch_clearing_price(A, B)
            if p is None:
                break
            
            final_p = p
            failed_orders = []
            
            for order in active_orders:
                if order.side == "X_TO_Y":
                    dy = order.amount_in * (1 - self.fee_rate) / p
                    exec_price = p
                    filled = dy >= order.min_out
                    if not filled:
                        failed_orders.append(order)
                    all_fills[order.order_id] = OrderFill(
                        order=order,
                        amount_out=dy,
                        execution_price=exec_price,
                        filled=filled,
                        fail_reason="" if filled else f"Output {dy:.4f} < minOut {order.min_out:.4f}"
                    )
                else:
                    dx = order.amount_in * (1 - self.fee_rate) * p
                    exec_price = p
                    filled = dx >= order.min_out
                    if not filled:
                        failed_orders.append(order)
                    all_fills[order.order_id] = OrderFill(
                        order=order,
                        amount_out=dx,
                        execution_price=exec_price,
                        filled=filled,
                        fail_reason="" if filled else f"Output {dx:.4f} < minOut {order.min_out:.4f}"
                    )
            
            if not failed_orders:
                delta_Y = A / p if p > 0 else 0
                delta_X = B * p
                
                final_X = initial_X + A - delta_X
                final_Y = initial_Y + B - delta_Y
                
                self.X = final_X
                self.Y = final_Y
                
                return BatchResult(
                    clearing_price=p,
                    final_X=final_X,
                    final_Y=final_Y,
                    delta_X=delta_X,
                    delta_Y=delta_Y,
                    fills=list(all_fills.values()),
                    initial_X=initial_X,
                    initial_Y=initial_Y,
                    iterations=iterations
                )
            
            active_orders = [o for o in active_orders if o not in failed_orders]
        
        return BatchResult(
            clearing_price=final_p,
            final_X=initial_X,
            final_Y=initial_Y,
            delta_X=0,
            delta_Y=0,
            fills=list(all_fills.values()),
            initial_X=initial_X,
            initial_Y=initial_Y,
            iterations=iterations
        )

def generate_random_orders(n_orders: int, base_amount: float, spread: float, min_out_ratio: float) -> List[Order]:
    orders = []
    for i in range(n_orders):
        side = np.random.choice(["X_TO_Y", "Y_TO_X"])
        amount = base_amount * (1 + np.random.uniform(-spread, spread))
        min_out = amount * min_out_ratio * np.random.uniform(0.3, 0.7)
        orders.append(Order(side=side, amount_in=amount, min_out=min_out, order_id=i+1))
    return orders

st.title("Batch Auction AMM Simulator")
st.markdown("""
This simulator implements a **constant-product AMM** with two execution modes:
1. **Sequential Mode**: Orders execute one-by-one, each affecting price for subsequent orders
2. **Batch Auction Mode**: All orders clear at a single uniform price, solving a quadratic equation

Compare how batch auctions can protect against MEV attacks by eliminating order-dependent price impact.
""")

st.divider()

st.header("Step 1: Pool Configuration")

col1, col2, col3 = st.columns(3)
with col1:
    reserve_x = st.number_input("Reserve X", min_value=1000.0, max_value=10000000.0, 
                                value=1000000.0, step=10000.0, format="%.0f")
with col2:
    reserve_y = st.number_input("Reserve Y", min_value=1000.0, max_value=10000000.0, 
                                value=1000000.0, step=10000.0, format="%.0f")
with col3:
    fee_rate = st.number_input("Fee Rate", min_value=0.0, max_value=0.1, 
                               value=0.003, step=0.001, format="%.4f")

k = reserve_x * reserve_y
spot_price = reserve_x / reserve_y

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Invariant k", f"{k:,.0f}")
with col2:
    st.metric("Spot Price (X/Y)", f"{spot_price:.6f}")
with col3:
    st.metric("Fee Rate", f"{fee_rate*100:.2f}%")

st.session_state['pool_config'] = {
    'reserve_x': reserve_x,
    'reserve_y': reserve_y,
    'fee_rate': fee_rate
}

st.divider()

st.header("Step 2: Order Entry")

if 'orders' not in st.session_state:
    st.session_state['orders'] = []

col1, col2 = st.columns(2)

with col1:
    st.subheader("Manual Order Entry")
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        new_side = st.selectbox("Side", ["X_TO_Y", "Y_TO_X"])
    with mcol2:
        new_amount = st.number_input("Amount In", min_value=0.01, value=10000.0, step=1000.0)
    with mcol3:
        new_min_out = st.number_input("Min Out", min_value=0.0, value=9000.0, step=100.0)
    
    if st.button("Add Order", type="primary"):
        order_id = len(st.session_state['orders']) + 1
        st.session_state['orders'].append(
            Order(side=new_side, amount_in=new_amount, min_out=new_min_out, order_id=order_id)
        )
        st.rerun()

with col2:
    st.subheader("Generate Random Orders")
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        n_random = st.number_input("Number of Orders", min_value=1, max_value=50, value=10)
        base_amount = st.number_input("Base Amount", min_value=100.0, value=10000.0, step=1000.0)
    with rcol2:
        amount_spread = st.slider("Amount Spread", 0.0, 1.0, 0.3)
        min_out_ratio = st.slider("Min Out Ratio", 0.0, 1.0, 0.5)
    
    if st.button("Generate Random Orders"):
        st.session_state['orders'] = generate_random_orders(n_random, base_amount, amount_spread, min_out_ratio)
        st.rerun()

st.subheader("Current Order Book")

if st.session_state['orders']:
    order_data = [o.to_dict() for o in st.session_state['orders']]
    df = pd.DataFrame(order_data)
    
    x_to_y = df[df['side'] == 'X_TO_Y']
    y_to_x = df[df['side'] == 'Y_TO_X']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders", len(df))
    with col2:
        st.metric("X→Y Orders", f"{len(x_to_y)} ({x_to_y['amountIn'].sum():,.0f} X)")
    with col3:
        st.metric("Y→X Orders", f"{len(y_to_x)} ({y_to_x['amountIn'].sum():,.0f} Y)")
    with col4:
        if st.button("Clear All Orders", type="secondary"):
            st.session_state['orders'] = []
            st.rerun()
    
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No orders in the batch. Add orders manually or generate random orders above.")

st.divider()

st.header("Step 3: Execute Both Modes")

if not st.session_state.get('orders'):
    st.warning("Please add orders first (Step 2) before executing.")
else:
    config = st.session_state['pool_config']
    orders = st.session_state['orders']
    
    if st.button("Execute Batch Auction & Sequential Mode", type="primary", use_container_width=True):
        amm_batch = ConstantProductAMM(config['reserve_x'], config['reserve_y'], config['fee_rate'])
        initial_spot = amm_batch.spot_price()
        batch_result = amm_batch.execute_batch(orders)
        
        amm_seq = ConstantProductAMM(config['reserve_x'], config['reserve_y'], config['fee_rate'])
        seq_initial_X, seq_initial_Y = amm_seq.X, amm_seq.Y
        
        execution_trace = []
        for i, order in enumerate(orders):
            price_before = amm_seq.spot_price()
            
            if order.side == "X_TO_Y":
                amount_out = amm_seq.swap_x_to_y(order.amount_in)
                exec_price = order.amount_in * (1 - config['fee_rate']) / amount_out if amount_out > 0 else 0
            else:
                amount_out = amm_seq.swap_y_to_x(order.amount_in)
                exec_price = amount_out / (order.amount_in * (1 - config['fee_rate'])) if order.amount_in > 0 else 0
            
            price_after = amm_seq.spot_price()
            
            execution_trace.append({
                "Step": i + 1,
                "Order ID": order.order_id,
                "Side": order.side,
                "Amount In": round(order.amount_in, 2),
                "Amount Out": round(amount_out, 4),
                "Exec Price": round(exec_price, 6),
                "Price Before": round(price_before, 6),
                "Price After": round(price_after, 6),
                "Price Impact": f"{((price_after - price_before) / price_before * 100):.4f}%",
                "Filled": "Yes" if amount_out >= order.min_out else "No"
            })
        
        seq_result = {
            'trace': execution_trace,
            'final_X': amm_seq.X,
            'final_Y': amm_seq.Y,
            'initial_X': seq_initial_X,
            'initial_Y': seq_initial_Y,
            'initial_spot': initial_spot,
            'final_spot': amm_seq.spot_price()
        }
        
        st.session_state['batch_result'] = batch_result
        st.session_state['seq_result'] = seq_result
        st.session_state['initial_spot'] = initial_spot
        
        st.success("Execution complete! See results below.")

if 'batch_result' in st.session_state and 'seq_result' in st.session_state:
    batch = st.session_state['batch_result']
    seq = st.session_state['seq_result']
    initial_spot = st.session_state['initial_spot']
    
    st.divider()
    
    st.header("Step 4: Results Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Batch Auction Results")
        
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.metric("Clearing Price (p)", f"{batch.clearing_price:.6f}")
            slippage = (batch.clearing_price - initial_spot) / initial_spot * 100
            st.metric("Price Impact", f"{slippage:.4f}%")
        with mcol2:
            st.metric("Iterations", batch.iterations)
            filled_count = sum(1 for f in batch.fills if f.filled)
            st.metric("Orders Filled", f"{filled_count} / {len(batch.fills)}")
        
        st.markdown("**Reserve Changes**")
        st.metric("X: Before → After", 
                 f"{batch.initial_X:,.2f} → {batch.final_X:,.2f}",
                 delta=f"{batch.final_X - batch.initial_X:,.2f}")
        st.metric("Y: Before → After", 
                 f"{batch.initial_Y:,.2f} → {batch.final_Y:,.2f}",
                 delta=f"{batch.final_Y - batch.initial_Y:,.2f}")
        
        k_before = batch.initial_X * batch.initial_Y
        k_after = batch.final_X * batch.final_Y
        st.metric("Invariant k Check", 
                 f"{k_before:,.0f} → {k_after:,.0f}",
                 delta=f"{((k_after/k_before)-1)*100:.6f}%")
    
    with col2:
        st.subheader("Sequential Mode Results")
        
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.metric("Final Price", f"{seq['final_spot']:.6f}")
            total_impact = (seq['final_spot'] - seq['initial_spot']) / seq['initial_spot'] * 100
            st.metric("Total Price Change", f"{total_impact:.4f}%")
        with mcol2:
            filled = sum(1 for t in seq['trace'] if t['Filled'] == "Yes")
            st.metric("Orders Filled", f"{filled} / {len(seq['trace'])}")
        
        st.markdown("**Reserve Changes**")
        st.metric("X: Before → After", 
                 f"{seq['initial_X']:,.2f} → {seq['final_X']:,.2f}",
                 delta=f"{seq['final_X'] - seq['initial_X']:,.2f}")
        st.metric("Y: Before → After", 
                 f"{seq['initial_Y']:,.2f} → {seq['final_Y']:,.2f}",
                 delta=f"{seq['final_Y'] - seq['initial_Y']:,.2f}")
    
    st.divider()
    
    st.header("Step 5: Detailed Analysis")
    
    st.subheader("Execution Mode Comparison")
    
    comparison_data = {
        "Metric": [
            "Clearing/Final Price",
            "Price Change from Initial",
            "Final Reserve X",
            "Final Reserve Y",
            "Orders Filled"
        ],
        "Batch Auction": [
            f"{batch.clearing_price:.6f}",
            f"{((batch.clearing_price - initial_spot) / initial_spot * 100):.4f}%",
            f"{batch.final_X:,.2f}",
            f"{batch.final_Y:,.2f}",
            f"{sum(1 for f in batch.fills if f.filled)} / {len(batch.fills)}"
        ],
        "Sequential": [
            f"{seq['final_spot']:.6f}",
            f"{((seq['final_spot'] - initial_spot) / initial_spot * 100):.4f}%",
            f"{seq['final_X']:,.2f}",
            f"{seq['final_Y']:,.2f}",
            f"{sum(1 for t in seq['trace'] if t['Filled'] == 'Yes')} / {len(seq['trace'])}"
        ]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.subheader("Price Uniformity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Batch Auction**: All orders execute at uniform clearing price")
        if batch.fills:
            batch_prices = [f.execution_price for f in batch.fills if f.filled]
            if batch_prices:
                bcol1, bcol2, bcol3 = st.columns(3)
                with bcol1:
                    st.metric("Min Exec Price", f"{min(batch_prices):.6f}")
                with bcol2:
                    st.metric("Max Exec Price", f"{max(batch_prices):.6f}")
                with bcol3:
                    st.metric("Price Variance", f"{np.var(batch_prices):.10f}")
    
    with col2:
        st.markdown("**Sequential Mode**: Each order gets different execution price")
        if seq['trace']:
            seq_prices = [t['Exec Price'] for t in seq['trace']]
            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                st.metric("Min Exec Price", f"{min(seq_prices):.6f}")
            with scol2:
                st.metric("Max Exec Price", f"{max(seq_prices):.6f}")
            with scol3:
                st.metric("Price Variance", f"{np.var(seq_prices):.10f}")
    
    st.subheader("MEV Protection Analysis")
    st.markdown("""
    **Key Insight**: In batch auction mode, all orders clear at the same price, 
    eliminating the ability for attackers to profit by ordering transactions.
    
    In sequential mode, the order of transactions matters significantly:
    - First orders get better prices
    - Later orders suffer from price impact
    - This creates MEV opportunities for sandwich attacks
    """)
    
    if seq['trace']:
        first_price = seq['trace'][0]['Exec Price']
        last_price = seq['trace'][-1]['Exec Price']
        price_diff = abs(last_price - first_price) / first_price * 100
        
        st.metric(
            "Sequential Price Difference (First vs Last Order)",
            f"{price_diff:.4f}%",
            delta="MEV opportunity" if price_diff > 0.1 else "Minimal"
        )
    
    st.subheader("Price Evolution (Sequential Mode)")
    
    if seq['trace']:
        prices = [seq['initial_spot']] + [t['Price After'] for t in seq['trace']]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(prices))),
            y=prices,
            mode='lines+markers',
            name='Spot Price',
            line=dict(color='blue', width=2)
        ))
        fig.add_hline(y=seq['initial_spot'], line_dash="dash", 
                     line_color="gray", annotation_text="Initial Price")
        fig.add_hline(y=batch.clearing_price, line_dash="dash", 
                     line_color="green", annotation_text="Batch Clearing Price")
        fig.update_layout(
            xaxis_title="Order Number",
            yaxis_title="Spot Price (X/Y)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Slippage Distribution")
    
    if batch.fills and seq['trace']:
        fig = go.Figure()
        
        batch_slippages = [(f.execution_price - initial_spot) / initial_spot * 100 
                          for f in batch.fills if f.filled]
        seq_slippages = [(t['Exec Price'] - initial_spot) / initial_spot * 100 
                        for t in seq['trace']]
        
        if batch_slippages:
            fig.add_trace(go.Box(y=batch_slippages, name="Batch Auction", 
                                marker_color='green'))
        if seq_slippages:
            fig.add_trace(go.Box(y=seq_slippages, name="Sequential", 
                                marker_color='red'))
        
        fig.update_layout(
            yaxis_title="Slippage from Initial Price (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Batch Auction Order Fills"):
        if batch.fills:
            fill_data = []
            for f in batch.fills:
                fill_data.append({
                    "Order ID": f.order.order_id,
                    "Side": f.order.side,
                    "Amount In": round(f.order.amount_in, 2),
                    "Min Out": round(f.order.min_out, 2),
                    "Amount Out": round(f.amount_out, 4),
                    "Exec Price": round(f.execution_price, 6),
                    "Filled": "Yes" if f.filled else "No",
                    "Fail Reason": f.fail_reason
                })
            st.dataframe(pd.DataFrame(fill_data), use_container_width=True, hide_index=True)
    
    with st.expander("View Sequential Execution Trace"):
        if seq['trace']:
            st.dataframe(pd.DataFrame(seq['trace']), use_container_width=True, hide_index=True)

st.sidebar.markdown("### About")
st.sidebar.markdown("""
**Batch Auction AMM Simulator**

Implements constant-product AMM with:
- Uniform-price batch clearing
- Sequential execution baseline
- MinOut constraint handling

Based on the mathematical specification for 
two-asset constant-product pools with 
invariant k = X · Y.
""")

st.sidebar.markdown("### Formula Reference")
st.sidebar.latex(r"k = X \cdot Y")
st.sidebar.latex(r"\Delta y = \frac{Y \cdot dx'}{X + dx'}")
st.sidebar.latex(r"p = \frac{X_1}{Y_1}")

//! Batch-auction constant-product AMM
//!
//! - Constant-product pool (k = X * Y)
//! - Sequential execution 
//! - Batch auction execution (uniform clearing price via quadratic solve)
//! - Iterative `min_out` filtering for batch execution

mod math;
mod types;

pub use crate::types::{BatchResult, ConstantProductAmm, Order, OrderFill, OrderSide};



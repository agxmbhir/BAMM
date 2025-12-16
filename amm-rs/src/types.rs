use crate::math::rel_err;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderSide {
    XToY,
    YToX,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Order {
    pub order_id: u64,
    pub side: OrderSide,
    pub amount_in: f64,
    pub min_out: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrderFill {
    pub order: Order,
    pub amount_out: f64,
    pub execution_price: f64,
    pub filled: bool,
    pub fail_reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchResult {
    pub clearing_price: f64,
    pub final_x: f64,
    pub final_y: f64,
    pub delta_x: f64,
    pub delta_y: f64,
    pub fills: Vec<OrderFill>,
    pub initial_x: f64,
    pub initial_y: f64,
    pub iterations: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantProductAmm {
    /// Reserve of token X
    pub x: f64,
    /// Reserve of token Y
    pub y: f64,
    /// Fee rate, e.g. 0.003
    pub fee_rate: f64,
    /// Initial invariant (kept constant in this model; fees are treated as taken out of the pool)
    pub k: f64,
}

impl ConstantProductAmm {
    pub fn new(reserve_x: f64, reserve_y: f64, fee_rate: f64) -> Self {
        Self {
            x: reserve_x,
            y: reserve_y,
            fee_rate,
            k: reserve_x * reserve_y,
        }
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    pub fn spot_price(&self) -> f64 {
        self.x / self.y
    }

    pub fn swap_x_to_y(&mut self, dx: f64) -> f64 {
        let dx_eff = dx * (1.0 - self.fee_rate);
        let dy = (self.y * dx_eff) / (self.x + dx_eff);
        self.x += dx_eff;
        self.y -= dy;
        dy
    }

    pub fn swap_y_to_x(&mut self, dy: f64) -> f64 {
        let dy_eff = dy * (1.0 - self.fee_rate);
        let dx = (self.x * dy_eff) / (self.y + dy_eff);
        self.y += dy_eff;
        self.x -= dx;
        dx
    }

    pub fn execute_sequential(&mut self, orders: &[Order]) -> Vec<OrderFill> {
        let mut fills = Vec::with_capacity(orders.len());
        for order in orders {
            match order.side {
                OrderSide::XToY => {
                    let amount_out = self.swap_x_to_y(order.amount_in);
                    let exec_price = if amount_out > 0.0 {
                        (order.amount_in * (1.0 - self.fee_rate)) / amount_out
                    } else {
                        0.0
                    };
                    let filled = amount_out >= order.min_out;
                    let fail_reason = if filled {
                        String::new()
                    } else {
                        format!(
                            "Output {:.4} < minOut {:.4}",
                            amount_out, order.min_out
                        )
                    };
                    fills.push(OrderFill {
                        order: order.clone(),
                        amount_out,
                        execution_price: exec_price,
                        filled,
                        fail_reason,
                    });
                }
                OrderSide::YToX => {
                    let amount_out = self.swap_y_to_x(order.amount_in);
                    let exec_price = if order.amount_in > 0.0 {
                        amount_out / (order.amount_in * (1.0 - self.fee_rate))
                    } else {
                        0.0
                    };
                    let filled = amount_out >= order.min_out;
                    let fail_reason = if filled {
                        String::new()
                    } else {
                        format!(
                            "Output {:.4} < minOut {:.4}",
                            amount_out, order.min_out
                        )
                    };
                    fills.push(OrderFill {
                        order: order.clone(),
                        amount_out,
                        execution_price: exec_price,
                        filled,
                        fail_reason,
                    });
                }
            }
        }
        fills
    }

    /// Solve for uniform clearing price `p` given aggregated effective inputs:
    /// - `A`: total effective X in from X→Y orders
    /// - `B`: total effective Y in from Y→X orders

    pub fn solve_batch_clearing_price(&self, a_in: f64, b_in: f64) -> Option<f64> {
        let x0 = self.x;
        let y0 = self.y;
        let k = self.k;

        if a_in == 0.0 && b_in == 0.0 {
            return Some(x0 / y0);
        }

        if b_in == 0.0 {
            let denom = (x0 + a_in) * y0 - k;
            if denom <= 0.0 {
                return None;
            }
            let p = (x0 + a_in) * a_in / denom;
            return if p > 0.0 { Some(p) } else { None };
        }

        if a_in == 0.0 {
            let denom = b_in * (y0 + b_in);
            if denom <= 0.0 {
                return None;
            }
            let p = (x0 * (y0 + b_in) - k) / denom;
            return if p > 0.0 { Some(p) } else { None };
        }

        // Quadratic: a*p^2 + b*p + c = 0
        let a = b_in * (y0 + b_in);
        let b = -(((x0 + a_in) * (y0 + b_in)) + (a_in * b_in) - k);
        let c = (x0 + a_in) * a_in;

        let disc = (b * b) - (4.0 * a * c);
        if disc < 0.0 {
            return None;
        }

        let sqrt_disc = disc.sqrt();
        let p1 = (-b + sqrt_disc) / (2.0 * a);
        let p2 = (-b - sqrt_disc) / (2.0 * a);

        #[derive(Clone, Copy)]
        struct Soln {
            p: f64,
            inv_err: f64,
            reserve_change: f64,
        }

        let mut sols: Vec<Soln> = Vec::new();
        for p in [p1, p2] {
            if p > 0.0 {
                let x1 = x0 + a_in - (b_in * p);
                let y1 = y0 + b_in - (a_in / p);
                if x1 > 0.0 && y1 > 0.0 {
                    let inv_err = rel_err(x1 * y1, k);
                    let reserve_change = (x1 - x0).abs() + (y1 - y0).abs();
                    sols.push(Soln {
                        p,
                        inv_err,
                        reserve_change,
                    });
                }
            }
        }

        if sols.is_empty() {
            return None;
        }

        let mut non_degenerate: Vec<Soln> = sols
            .iter()
            .copied()
            .filter(|s| s.reserve_change > 0.01)
            .collect();

        if !non_degenerate.is_empty() {
            non_degenerate.sort_by(|a, b| a.inv_err.partial_cmp(&b.inv_err).unwrap());
            return Some(non_degenerate[0].p);
        }

        sols.sort_by(|a, b| a.inv_err.partial_cmp(&b.inv_err).unwrap());
        Some(sols[0].p)
    }

    /// Execute the batch auction with iterative `min_out` filtering (matches `app.py` behavior).
    pub fn execute_batch(&mut self, orders: &[Order]) -> BatchResult {
        let initial_x = self.x;
        let initial_y = self.y;

        let mut active: Vec<Order> = orders.to_vec();
        let mut fills: Vec<OrderFill> = orders
            .iter()
            .cloned()
            .map(|o| OrderFill {
                order: o,
                amount_out: 0.0,
                execution_price: 0.0,
                filled: false,
                fail_reason: "Not processed".to_string(),
            })
            .collect();

        let mut iterations: u32 = 0;
        let max_iterations: u32 = 100;
        let mut final_p = initial_x / initial_y;

        while iterations < max_iterations {
            iterations += 1;

            let a_in: f64 = active
                .iter()
                .filter(|o| o.side == OrderSide::XToY)
                .map(|o| o.amount_in * (1.0 - self.fee_rate))
                .sum();
            let b_in: f64 = active
                .iter()
                .filter(|o| o.side == OrderSide::YToX)
                .map(|o| o.amount_in * (1.0 - self.fee_rate))
                .sum();

            if a_in == 0.0 && b_in == 0.0 {
                // Pool unchanged; annotate removed orders to match the Python app.
                for fill in &mut fills {
                    let still_active = active.iter().any(|o| o.order_id == fill.order.order_id);
                    if !still_active {
                        *fill = OrderFill {
                            order: fill.order.clone(),
                            amount_out: 0.0,
                            execution_price: 0.0,
                            filled: false,
                            fail_reason: "Removed in minOut filtering".to_string(),
                        };
                    }
                }

                return BatchResult {
                    clearing_price: final_p,
                    final_x: initial_x,
                    final_y: initial_y,
                    delta_x: 0.0,
                    delta_y: 0.0,
                    fills,
                    initial_x,
                    initial_y,
                    iterations,
                };
            }

            let p = match self.solve_batch_clearing_price(a_in, b_in) {
                Some(p) => p,
                None => break,
            };
            final_p = p;

            let mut failed_order_ids: Vec<u64> = Vec::new();

            // Update fills for active orders, leave removed orders as-is (matches Python).
            for order in &active {
                match order.side {
                    OrderSide::XToY => {
                        let out = (order.amount_in * (1.0 - self.fee_rate)) / p;
                        let filled_ok = out >= order.min_out;
                        if !filled_ok {
                            failed_order_ids.push(order.order_id);
                        }
                        if let Some(fill) = fills.iter_mut().find(|f| f.order.order_id == order.order_id)
                        {
                            *fill = OrderFill {
                                order: order.clone(),
                                amount_out: out,
                                execution_price: p,
                                filled: filled_ok,
                                fail_reason: if filled_ok {
                                    String::new()
                                } else {
                                    format!("Output {:.4} < minOut {:.4}", out, order.min_out)
                                },
                            };
                        }
                    }
                    OrderSide::YToX => {
                        let out = (order.amount_in * (1.0 - self.fee_rate)) * p;
                        let filled_ok = out >= order.min_out;
                        if !filled_ok {
                            failed_order_ids.push(order.order_id);
                        }
                        if let Some(fill) = fills.iter_mut().find(|f| f.order.order_id == order.order_id)
                        {
                            *fill = OrderFill {
                                order: order.clone(),
                                amount_out: out,
                                execution_price: p,
                                filled: filled_ok,
                                fail_reason: if filled_ok {
                                    String::new()
                                } else {
                                    format!("Output {:.4} < minOut {:.4}", out, order.min_out)
                                },
                            };
                        }
                    }
                }
            }

            if failed_order_ids.is_empty() {
                let delta_y = if p > 0.0 { a_in / p } else { 0.0 };
                let delta_x = b_in * p;
                let final_x = initial_x + a_in - delta_x;
                let final_y = initial_y + b_in - delta_y;

                self.x = final_x;
                self.y = final_y;

                return BatchResult {
                    clearing_price: p,
                    final_x,
                    final_y,
                    delta_x,
                    delta_y,
                    fills,
                    initial_x,
                    initial_y,
                    iterations,
                };
            }

            active.retain(|o| !failed_order_ids.contains(&o.order_id));
        }

        BatchResult {
            clearing_price: final_p,
            final_x: initial_x,
            final_y: initial_y,
            delta_x: 0.0,
            delta_y: 0.0,
            fills,
            initial_x,
            initial_y,
            iterations,
        }
    }
}



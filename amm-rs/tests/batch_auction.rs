use batch_auction_amm::{ConstantProductAmm, Order, OrderSide};

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

fn rel_err(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        if a == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        ((a - b).abs()) / b.abs()
    }
}

#[test]
fn spot_price_when_no_orders() {
    let amm = ConstantProductAmm::new(1_000_000.0, 2_000_000.0, 0.003);
    let p = amm.solve_batch_clearing_price(0.0, 0.0).unwrap();
    assert!(approx_eq(p, amm.spot_price(), 1e-12));
}

#[test]
fn batch_invariant_is_preserved_on_success() {
    let mut amm = ConstantProductAmm::new(1_000_000.0, 1_000_000.0, 0.003);
    let orders = vec![
        Order {
            order_id: 1,
            side: OrderSide::XToY,
            amount_in: 50_000.0,
            min_out: 0.0,
        },
        Order {
            order_id: 2,
            side: OrderSide::YToX,
            amount_in: 10_000.0,
            min_out: 0.0,
        },
    ];

    let k0 = amm.k;
    let res = amm.execute_batch(&orders);
    let k1 = res.final_x * res.final_y;
    assert!(rel_err(k1, k0) < 1e-9, "k drift too large: {}", rel_err(k1, k0));
    assert!(res.delta_x > 0.0 || res.delta_y > 0.0);
    assert!(res.clearing_price > 0.0);
}

#[test]
fn min_out_filtering_removes_failing_orders() {
    let mut amm = ConstantProductAmm::new(1_000_000.0, 1_000_000.0, 0.003);
    let too_strict = Order {
        order_id: 1,
        side: OrderSide::XToY,
        amount_in: 1000.0,
        min_out: 1_000_000_000.0,
    };
    let ok = Order {
        order_id: 2,
        side: OrderSide::YToX,
        amount_in: 2000.0,
        min_out: 0.0,
    };
    let res = amm.execute_batch(&[too_strict.clone(), ok.clone()]);

    let fill1 = res.fills.iter().find(|f| f.order.order_id == 1).unwrap();
    assert!(!fill1.filled);
    assert!(fill1.fail_reason.contains("minOut"));

    let fill2 = res.fills.iter().find(|f| f.order.order_id == 2).unwrap();
    // Should still be processed (maybe even succeeds in later iteration).
    assert!(fill2.fail_reason.is_empty() || fill2.fail_reason.contains("minOut"));
}

#[test]
fn sequential_executes_each_order_and_updates_reserves() {
    let mut amm = ConstantProductAmm::new(1_000_000.0, 1_000_000.0, 0.003);
    let orders = vec![
        Order {
            order_id: 1,
            side: OrderSide::XToY,
            amount_in: 10_000.0,
            min_out: 0.0,
        },
        Order {
            order_id: 2,
            side: OrderSide::YToX,
            amount_in: 10_000.0,
            min_out: 0.0,
        },
    ];
    let x0 = amm.x;
    let y0 = amm.y;
    let fills = amm.execute_sequential(&orders);
    assert_eq!(fills.len(), 2);
    assert!((amm.x - x0).abs() > 0.0 || (amm.y - y0).abs() > 0.0);
}

#[test]
fn batch_matches_sequential_for_single_order_when_min_out_allows() {
    // With a single order, batch price resolves to the same post-trade state as sequential.
    let orders = vec![Order {
        order_id: 1,
        side: OrderSide::XToY,
        amount_in: 25_000.0,
        min_out: 0.0,
    }];

    let mut seq = ConstantProductAmm::new(1_000_000.0, 1_000_000.0, 0.003);
    let seq_fills = seq.execute_sequential(&orders);
    let seq_out = seq_fills[0].amount_out;

    let mut batch = ConstantProductAmm::new(1_000_000.0, 1_000_000.0, 0.003);
    let batch_res = batch.execute_batch(&orders);
    let batch_out = batch_res
        .fills
        .iter()
        .find(|f| f.order.order_id == 1)
        .unwrap()
        .amount_out;

    assert!(approx_eq(batch_out, seq_out, 1e-9));
    assert!(approx_eq(batch.x, seq.x, 1e-9));
    assert!(approx_eq(batch.y, seq.y, 1e-9));
}



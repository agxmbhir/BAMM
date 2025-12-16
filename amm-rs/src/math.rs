pub fn rel_err(a: f64, b: f64) -> f64 {
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



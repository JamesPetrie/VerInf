// Copyright (c) 2026 Levi Neuwirth. Released under the MIT License (see LICENSE).
//! Run the row rule on a real statement without parsing its proof, which at full
//! scale takes tens of GB.
//!
//! Usage: layout_check statement.json blind=3 w=1974800 p1=8273657 p2=3851796 p3=216
//! The blocks are given in the statement's order.
use ligero_verifier::claim::parse_claim_set;
use ligero_verifier::layout::check_rows;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cs = parse_claim_set(&std::fs::read_to_string(&args[0]).expect("read the statement"));
    let (mut order, mut rows) = (Vec::new(), Vec::new());
    for a in &args[1..] {
        let (name, n) = a.split_once('=').expect("block=rows");
        order.push(name.to_string());
        rows.push(n.parse::<usize>().expect("a row count"));
    }
    match check_rows(&cs, &order, &rows) {
        Ok(()) => println!("rows OK: {} claims, {} rows", cs.claims.len(), rows.iter().sum::<usize>()),
        Err(e) => { println!("rows REJECT: {e}"); std::process::exit(1); }
    }
}

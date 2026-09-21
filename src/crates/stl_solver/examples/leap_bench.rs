//! We benchmark a packing LP for M = f*11^T - d*A, d > 0.
//! We use a primal or dual simplex dictionary and certify the original game.
//! We retain no values in the production sweep. Python solves rejected stages.
use rayon::prelude::*;
use std::{env, fs, time::Instant};
#[path = "../src/leap_packing.rs"]
mod packing;
use packing::Solver;

fn main() {
    let args: Vec<_> = env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    assert_eq!(bytes.len() % (62 * 8), 0);
    let input: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let warm = args[3] == "1";
    rayon::ThreadPoolBuilder::new()
        .num_threads(args[4].parse().unwrap())
        .build_global()
        .unwrap();
    let mut output = vec![[0.; 5]; input.len() / 62];
    let tick = Instant::now();
    output
        .par_chunks_mut(2048)
        .enumerate()
        .for_each(|(chunk, rows)| {
            let mut solver = Solver::new();
            let mut group = f64::NAN;
            for (i, row) in rows.iter_mut().enumerate() {
                let pos = (chunk * 2048 + i) * 62;
                let stage = &input[pos..pos + 62];
                if stage[61] != group {
                    solver.ready = false;
                    group = stage[61];
                }
                *row = solver.solve(stage, warm);
            }
        });
    let seconds = tick.elapsed().as_secs_f64();
    let failed = output.iter().filter(|x| !x[0].is_finite()).count();
    let data: Vec<u8> = output
        .iter()
        .flatten()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    fs::write(&args[2], data).unwrap();
    println!("{{\"solve_seconds\":{seconds},\"failed\":{failed}}}");
}

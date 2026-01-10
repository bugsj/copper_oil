use std::env;

use copper_oil::read_and_plot_data;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args();
    let cmd = args.next().expect("args error!?");
    if args.len() < 1 {
        eprintln!("Usage: {} <input_file>", cmd);
        std::process::exit(1);
    }
    read_and_plot_data(args).inspect_err(|e| { eprintln!("error! {}", e); })
}

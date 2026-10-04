use clap::Parser;

#[derive(Parser)]
#[command(
    name = "dictation",
    version,
    about = "Push-to-talk dictation with whisper-burn"
)]
struct Args {}

fn main() {
    let _ = Args::parse();
}

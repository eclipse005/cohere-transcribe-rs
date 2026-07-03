fn main() -> anyhow::Result<()> {
    // ASR inference can build deep stacks on a 16 MiB default; match the parent
    // CLI's generous stack so encoder/decoder recursion never overflows.
    std::thread::Builder::new()
        .name("native-transcribe".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(native_transcribe::run_cli)?
        .join()
        .map_err(|_| anyhow::anyhow!("CLI thread panicked"))?
}

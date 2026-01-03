//! Shell completion generation for TensorLogic CLI

use clap::CommandFactory;
use clap_complete::{generate, Generator, Shell};
use std::io;

use crate::cli::Cli;

#[allow(dead_code)]
pub fn generate_completion<G: Generator>(gen: G) {
    let mut cmd = Cli::command();
    generate(gen, &mut cmd, "tensorlogic", &mut io::stdout());
}

#[allow(dead_code)]
pub fn generate_for_shell(shell: Shell) {
    match shell {
        Shell::Bash => generate_completion(Shell::Bash),
        Shell::Zsh => generate_completion(Shell::Zsh),
        Shell::Fish => generate_completion(Shell::Fish),
        Shell::PowerShell => generate_completion(Shell::PowerShell),
        _ => eprintln!("Unsupported shell"),
    }
}

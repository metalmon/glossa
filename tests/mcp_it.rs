use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn mcp_subcommand_exists_with_profile_flag() {
    Command::cargo_bin("kb").unwrap()
        .args(["mcp", "--help"])
        .assert().success()
        .stdout(contains("--profile"));
}

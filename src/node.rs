//! Slave / worker runtime (band B7/B8): a `rusty` binary placed on a local
//! host (Windows/Linux via Rust-docker, or Android via cargo-ndk/Termux .apk/.bin)
//! that runs inference and reports to the GSV-native master supervisor.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Master,
    Slave,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::Master => f.write_str("master"),
            Role::Slave => f.write_str("slave"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_display() {
        assert_eq!(Role::Master.to_string(), "master");
        assert_eq!(Role::Slave.to_string(), "slave");
    }
}

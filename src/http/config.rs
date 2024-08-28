pub struct Config {
    pub port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self { port: 18000 }
    }
}

impl Config {
    pub fn new(port: u16) -> Self {
        Self { port }
    }
}

use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use anyhow;

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub(crate) struct ConfigFile {
    pub tari_address: String,
    pub tari_node_url: String,
    pub coinbase_extra: String,
    pub template_refresh_secs: u64,
    pub p2pool_enabled: bool,
    pub http_server_enabled: bool,
    pub http_server_port: u16,
    pub gpu_percentage: u16,
    pub grid_size: u32,
    pub template_timeout_secs: u64,
}

impl Default for ConfigFile {
    fn default() -> Self {
        Self {
            tari_address: "f2CWXg4GRNXweuDknxLATNjeX8GyJyQp9GbVG8f81q63hC7eLJ4ZR8cDd9HBcVTjzoHYUtzWZFM3yrZ68btM2wiY7sj"
                .to_string(),
            tari_node_url: "http://127.0.0.1:18142".to_string(),
            coinbase_extra: "tari_gpu_miner".to_string(),
            template_refresh_secs: 30,
            p2pool_enabled: false,
            http_server_enabled: true,
            http_server_port: 18000,
            // In range 1-1000
            gpu_percentage: 1000,
            grid_size: 1000,
            template_timeout_secs: 1,
        }
    }
}

impl ConfigFile {
    pub(crate) fn load(path: &PathBuf) -> Result<Self, anyhow::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }

    pub(crate) fn save(&self, path: &Path) -> Result<(), anyhow::Error> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

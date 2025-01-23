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
    pub block_size: u32,
    pub single_grid_size: u32,
    pub per_device_grid_sizes: Vec<u32>,
    pub template_timeout_secs: u64,
    #[serde(default = "default_max_template_failures")]
    pub max_template_failures: u64,
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
            block_size: 896,
            single_grid_size: 1024,
            per_device_grid_sizes: vec![],
            template_timeout_secs: 1,
            max_template_failures: 10,
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

fn default_max_template_failures() -> u64 {
    10
}

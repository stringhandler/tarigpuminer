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
    pub gpu_percentage: u8,
}

impl Default for ConfigFile {
    fn default() -> Self {
        Self {
            tari_address: "8c98d40f216589d8b385015222b95fb5327fee334352c7c30370101b0c6d124fd6".to_string(),
            tari_node_url: "http://127.0.0.1:18142".to_string(),
            coinbase_extra: "tari_gpu_miner".to_string(),
            template_refresh_secs: 30,
            p2pool_enabled: false,
            http_server_enabled: true,
            http_server_port: 18000,
            gpu_percentage: 100,
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

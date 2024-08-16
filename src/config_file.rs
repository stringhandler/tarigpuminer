use std::{fs::File, io::BufReader, path::PathBuf};

use anyhow;

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub(crate) struct ConfigFile {
    pub tari_address: String,
    pub tari_node_url: String,
    pub coinbase_extra: String,
    pub template_refresh_secs: u64,
    pub use_p2pool: bool,
}

impl Default for ConfigFile {
    fn default() -> Self {
        Self {
            tari_address: "8c98d40f216589d8b385015222b95fb5327fee334352c7c30370101b0c6d124fd6".to_string(),
            tari_node_url: "http://127.0.0.1:18142".to_string(),
            coinbase_extra: "tari_gpu_miner".to_string(),
            template_refresh_secs: 30,
            use_p2pool: false,
        }
    }
}

impl ConfigFile {
    pub(crate) fn load(path: PathBuf) -> Result<Self, anyhow::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }

    pub(crate) fn save(&self, path: &str) -> Result<(), anyhow::Error> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

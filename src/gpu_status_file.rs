use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use anyhow;

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct GpuStatus {
    pub device_index: u32,
    pub device_name: String,
    pub is_available: bool,
    pub is_excluded: bool,
    pub grid_size: u32,
    pub max_grid_size: u32,
    pub block_size: u32,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct GpuStatusFile {
    pub gpu_devices: Vec<GpuStatus>,
}

impl Default for GpuStatusFile {
    fn default() -> Self {
        Self { gpu_devices: vec![] }
    }
}

impl GpuStatusFile {
    pub fn new(gpu_devices: Vec<GpuStatus>) -> Self {
        Self { gpu_devices }
    }

    pub fn load(path: &PathBuf) -> Result<Self, anyhow::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<(), anyhow::Error> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

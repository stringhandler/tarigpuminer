use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use anyhow;

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub(crate) struct GpuStatusFile {
    pub num_devices: u32,
    pub device_names: Vec<String>,
}

impl Default for GpuStatusFile {
    fn default() -> Self {
        Self {
            num_devices: 0,
            device_names: vec![],
        }
    }
}

impl GpuStatusFile {
    pub const fn new(num_devices: u32, device_names: Vec<String>) -> Self {
        Self {
            num_devices,
            device_names,
        }
    }

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

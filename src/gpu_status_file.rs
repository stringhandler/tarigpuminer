use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use anyhow;

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct GpuStatus {
    pub device_index: u32,
    pub device_name: String,
    pub recommended_grid_size: u32,
    pub recommended_block_size: u32,
    pub max_grid_size: u32,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct GpuSettings {
    pub device_index: u32,
    pub device_name: String,
    pub is_excluded: bool,
    pub is_available: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct GpuStatusFile {
    pub gpu_devices_statuses: Vec<GpuStatus>,
    pub gpu_devices_settings: Vec<GpuSettings>,
}

impl Default for GpuStatusFile {
    fn default() -> Self {
        Self {
            gpu_devices_statuses: vec![],
            gpu_devices_settings: vec![],
        }
    }
}

impl GpuStatusFile {
    pub fn new(gpu_devices: Vec<GpuStatus>, file_path: &PathBuf) -> Self {
        let resolved_gpu_file = Self::resolve_settings_for_detected_devices(gpu_devices, file_path);

        resolved_gpu_file
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

    fn resolve_settings_for_detected_devices(gpu_devices: Vec<GpuStatus>, file_path: &PathBuf) -> GpuStatusFile {
        let mut resolved_gpu_file: GpuStatusFile = GpuStatusFile {
            gpu_devices_statuses: gpu_devices.clone(),
            gpu_devices_settings: vec![],
        };

        resolved_gpu_file.gpu_devices_settings = gpu_devices
            .into_iter()
            .map(|gpu_device| match Self::load(file_path) {
                Ok(gpu_file) => {
                    let gpu_settings = gpu_file
                        .gpu_devices_settings
                        .into_iter()
                        .find(|gpu_setting| gpu_setting.device_name == gpu_device.device_name);
                    match gpu_settings {
                        Some(gpu_setting) => GpuSettings {
                            device_name: gpu_device.device_name,
                            device_index: gpu_device.device_index,
                            is_excluded: gpu_setting.is_excluded,
                            is_available: gpu_setting.is_available,
                        },
                        None => GpuSettings {
                            device_name: gpu_device.device_name,
                            device_index: gpu_device.device_index,
                            is_excluded: false,
                            is_available: true,
                        },
                    }
                },
                Err(_) => GpuSettings {
                    device_name: gpu_device.device_name,
                    device_index: gpu_device.device_index,
                    is_excluded: false,
                    is_available: true,
                },
            })
            .collect();

        return resolved_gpu_file;
    }
}

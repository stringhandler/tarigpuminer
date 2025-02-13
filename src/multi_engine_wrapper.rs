use std::{any::Any, fs::create_dir_all, path::PathBuf};

use log::warn;

#[cfg(feature = "nvidia")]
use crate::cuda_engine::CudaEngine;
#[cfg(feature = "metal")]
use crate::metal_engine::MetalEngine;
#[cfg(feature = "opencl")]
use crate::opencl_engine::OpenClEngine;
use crate::{
    engine_impl::EngineImpl,
    gpu_engine::GpuEngine,
    gpu_status_file::{GpuDevice, GpuStatus, GpuStatusFile},
};

const LOG_TARGET: &str = "tari::gpuminer::multi_engine_wrapper";

#[derive(Debug, PartialEq, Clone)]
pub enum EngineType {
    Cuda,
    OpenCL,
    Metal,
}

impl EngineType {
    pub fn to_string(&self) -> String {
        match self {
            EngineType::Cuda => "CUDA".to_string(),
            EngineType::OpenCL => "OpenCL".to_string(),
            EngineType::Metal => "Metal".to_string(),
        }
    }

    pub fn from_string(engine_type: &str) -> Self {
        match engine_type {
            "CUDA" => EngineType::Cuda,
            "OpenCL" => EngineType::OpenCL,
            "Metal" => EngineType::Metal,
            _ => panic!("Unknown engine type"),
        }
    }
}

#[derive(Clone)]
pub struct MultiEngineWrapper {
    selected_engine: EngineType,
    #[cfg(feature = "nvidia")]
    cuda_engine: GpuEngine<CudaEngine>,
    #[cfg(feature = "opencl")]
    opencl_engine: GpuEngine<OpenClEngine>,
    #[cfg(feature = "metal")]
    metal_engine: GpuEngine<MetalEngine>,
}

impl MultiEngineWrapper {
    pub fn new(selected_engine: EngineType) -> Self {
        Self {
            selected_engine,
            #[cfg(feature = "nvidia")]
            cuda_engine: GpuEngine::new(CudaEngine::new()),
            #[cfg(feature = "opencl")]
            opencl_engine: GpuEngine::new(OpenClEngine::new()),
            #[cfg(feature = "metal")]
            metal_engine: GpuEngine::new(MetalEngine::new()),
        }
    }

    pub fn create_status_file(
        &self,
        destination_folder: &PathBuf,
        engine_type: EngineType,
        gpu_devices: Vec<GpuDevice>,
    ) -> Result<(), anyhow::Error> {
        let file_name = format!("{}_gpu_status.json", engine_type.to_string());
        let status_file_path = destination_folder.join(file_name);

        let status_file = GpuStatusFile::new(gpu_devices, &status_file_path);

        let _ = match GpuStatusFile::load(&status_file_path) {
            Ok(_) => {
                if let Err(err) = status_file.save(&status_file_path) {
                    warn!(target: LOG_TARGET,"Error saving gpu status: {}", err);
                }
                status_file
            },
            Err(_) => {
                if let Err(err) = create_dir_all(&status_file_path.parent().expect("no parent")) {
                    warn!(target: LOG_TARGET, "Error creating directory: {}", err);
                }
                if let Err(err) = status_file.save(&status_file_path) {
                    warn!(target: LOG_TARGET,"Error saving gpu status: {}", err);
                }
                status_file
            },
        };

        return Ok(());
    }

    pub fn create_status_files_for_each_engine(&mut self, destination_folder: PathBuf) -> Vec<EngineType> {
        let mut engines_with_created_status_files: Vec<EngineType> = Vec::new();
        #[cfg(feature = "opencl")]
        {
            if let Err(err) = self.opencl_engine.init() {
                warn!(target: LOG_TARGET, "Error initializing OpenCL engine: {}", err);
            }
            match self.opencl_engine.detect_devices() {
                Ok(gpu_devices) => {
                    if let Ok(_) = self.create_status_file(&destination_folder, EngineType::OpenCL, gpu_devices) {
                        engines_with_created_status_files.push(EngineType::OpenCL);
                    }
                },
                Err(err) => {
                    warn!(target: LOG_TARGET, "Error detecting OpenCL devices: {}", err);
                },
            };
        }

        #[cfg(feature = "nvidia")]
        {
            if let Err(err) = self.cuda_engine.init() {
                warn!(target: LOG_TARGET, "Error initializing CUDA engine: {}", err);
            }
            match self.cuda_engine.detect_devices() {
                Ok(gpu_devices) => {
                    if let Ok(_) = self.create_status_file(&destination_folder, EngineType::Cuda, gpu_devices) {
                        engines_with_created_status_files.push(EngineType::Cuda);
                    }
                },
                Err(err) => {
                    warn!(target: LOG_TARGET, "Error detecting CUDA devices: {}", err);
                },
            };
        }

        #[cfg(feature = "metal")]
        {
            if let Err(err) = self.metal_engine.init() {
                warn!(target: LOG_TARGET, "Error initializing Metal engine: {}", err);
            }
            match self.metal_engine.detect_devices() {
                Ok(gpu_devices) => {
                    if let Ok(_) = self.create_status_file(&destination_folder, EngineType::Metal, gpu_devices) {
                        engines_with_created_status_files.push(EngineType::Metal);
                    }
                },
                Err(err) => {
                    warn!(target: LOG_TARGET, "Error detecting Metal devices: {}", err);
                },
            };
        }

        return engines_with_created_status_files;
    }
}

impl EngineImpl for MultiEngineWrapper {
    type Context = Box<dyn Any>;
    type Function = Box<dyn Any>;

    fn get_engine_type(&self) -> EngineType {
        self.selected_engine.clone()
    }

    fn init(&mut self) -> Result<(), anyhow::Error> {
        match self.selected_engine {
            #[cfg(feature = "nvidia")]
            EngineType::Cuda => self.cuda_engine.init(),
            #[cfg(feature = "opencl")]
            EngineType::OpenCL => self.opencl_engine.init(),
            #[cfg(feature = "metal")]
            EngineType::Metal => self.metal_engine.init(),
            _ => panic!("Unknown engine type"),
        }
    }

    fn num_devices(&self) -> Result<u32, anyhow::Error> {
        match self.selected_engine {
            #[cfg(feature = "nvidia")]
            EngineType::Cuda => self.cuda_engine.num_devices(),
            #[cfg(feature = "opencl")]
            EngineType::OpenCL => self.opencl_engine.num_devices(),
            #[cfg(feature = "metal")]
            EngineType::Metal => self.metal_engine.num_devices(),
            _ => panic!("Unknown engine type"),
        }
    }

    fn detect_devices(&self) -> Result<Vec<GpuDevice>, anyhow::Error> {
        match self.selected_engine {
            #[cfg(feature = "nvidia")]
            EngineType::Cuda => self.cuda_engine.detect_devices(),
            #[cfg(feature = "opencl")]
            EngineType::OpenCL => self.opencl_engine.detect_devices(),
            #[cfg(feature = "metal")]
            EngineType::Metal => self.metal_engine.detect_devices(),
            _ => panic!("Unknown engine type"),
        }
    }

    fn create_context(&self, device_index: u32) -> Result<Self::Context, anyhow::Error> {
        match self.selected_engine {
            #[cfg(feature = "nvidia")]
            EngineType::Cuda => self
                .cuda_engine
                .create_context(device_index)
                .map(|f| Box::new(f) as Box<dyn Any>),
            #[cfg(feature = "opencl")]
            EngineType::OpenCL => self
                .opencl_engine
                .create_context(device_index)
                .map(|f| Box::new(f) as Box<dyn Any>),
            #[cfg(feature = "metal")]
            EngineType::Metal => self
                .metal_engine
                .create_context(device_index)
                .map(|f| Box::new(f) as Box<dyn Any>),
            _ => panic!("Unknown engine type"),
        }
    }

    fn create_main_function(&self, context: &Self::Context) -> Result<Self::Function, anyhow::Error> {
        match self.selected_engine {
            #[cfg(feature = "nvidia")]
            EngineType::Cuda => self
                .cuda_engine
                .get_main_function(context.downcast_ref().unwrap())
                .map(|f| Box::new(f) as Box<dyn Any>),
            #[cfg(feature = "opencl")]
            EngineType::OpenCL => self
                .opencl_engine
                .get_main_function(context.downcast_ref().unwrap())
                .map(|f| Box::new(f) as Box<dyn Any>),
            #[cfg(feature = "metal")]
            EngineType::Metal => self
                .metal_engine
                .get_main_function(context.downcast_ref().unwrap())
                .map(|f| Box::new(f) as Box<dyn Any>),
            _ => panic!("Unknown engine type"),
        }
    }

    fn mine(
        &self,
        function: &Self::Function,
        context: &Self::Context,
        data: &[u64],
        min_difficulty: u64,
        nonce_start: u64,
        num_iterations: u32,
        block_size: u32,
        grid_size: u32,
    ) -> Result<(Option<u64>, u32, u64), anyhow::Error> {
        match self.selected_engine {
            #[cfg(feature = "nvidia")]
            EngineType::Cuda => self.cuda_engine.mine(
                function.downcast_ref().unwrap(),
                context.downcast_ref().unwrap(),
                data,
                min_difficulty,
                nonce_start,
                num_iterations,
                block_size,
                grid_size,
            ),
            #[cfg(feature = "opencl")]
            EngineType::OpenCL => self.opencl_engine.mine(
                function.downcast_ref().unwrap(),
                context.downcast_ref().unwrap(),
                data,
                min_difficulty,
                nonce_start,
                num_iterations,
                block_size,
                grid_size,
            ),
            #[cfg(feature = "metal")]
            EngineType::Metal => self.metal_engine.mine(
                function.downcast_ref().unwrap(),
                context.downcast_ref().unwrap(),
                data,
                min_difficulty,
                nonce_start,
                num_iterations,
                block_size,
                grid_size,
            ),
            _ => panic!("Unknown engine type"),
        }
    }
}

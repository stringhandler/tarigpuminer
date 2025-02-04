use std::{
    any::{self, Any},
    collections::HashMap,
};

use anyhow::anyhow;

#[cfg(feature = "nvidia")]
use crate::cuda_engine::CudaEngine;
#[cfg(feature = "metal")]
use crate::metal_engine::MetalEngine;
#[cfg(feature = "opencl3")]
use crate::opencl_engine::OpenClEngine;
use crate::{
    context_impl::ContextImpl,
    engine_impl::EngineImpl,
    function_impl::FunctionImpl,
    gpu_engine::GpuEngine,
    gpu_status_file::GpuStatus,
    opencl_engine::OpenClEngine,
};

enum EnginesEnum<CudaEngine: EngineImpl, OpenClEngine: EngineImpl, MetalEngine: EngineImpl> {
    Cuda(GpuEngine<CudaEngine>),
    OpenCL(GpuEngine<OpenClEngine>),
    Metal(GpuEngine<MetalEngine>),
}

impl<CudaEngine, OpenClEngine, MetalEngine> EngineImpl for EnginesEnum<CudaEngine, OpenClEngine, MetalEngine>
where
    CudaEngine: EngineImpl,
    OpenClEngine: EngineImpl,
    MetalEngine: EngineImpl,
{
    type Context = Box<dyn Any>;
    type Function = Box<dyn Any>;

    fn get_engine_type(&self) -> EngineType {
        match self {
            EnginesEnum::Cuda(_) => EngineType::Cuda,
            EnginesEnum::OpenCL(_) => EngineType::OpenCL,
            EnginesEnum::Metal(_) => EngineType::Metal,
        }
    }

    fn init(&mut self) -> Result<(), anyhow::Error> {
        match self {
            EnginesEnum::Cuda(engine) => engine.init(),
            EnginesEnum::OpenCL(engine) => engine.init(),
            EnginesEnum::Metal(engine) => engine.init(),
        }
    }

    fn num_devices(&self) -> Result<u32, anyhow::Error> {
        match self {
            EnginesEnum::Cuda(engine) => engine.num_devices(),
            EnginesEnum::OpenCL(engine) => engine.num_devices(),
            EnginesEnum::Metal(engine) => engine.num_devices(),
        }
    }

    fn detect_devices(&self) -> Result<Vec<GpuStatus>, anyhow::Error> {
        match self {
            EnginesEnum::Cuda(engine) => engine.detect_devices(),
            EnginesEnum::OpenCL(engine) => engine.detect_devices(),
            EnginesEnum::Metal(engine) => engine.detect_devices(),
        }
    }

    fn create_main_function(&self, context: &Self::Context) -> Result<Self::Function, anyhow::Error> {
        match self {
            EnginesEnum::Cuda(engine) => engine
                .get_main_function(context.downcast_ref().unwrap())
                .map(|f| Box::new(f) as Box<dyn Any>),
            EnginesEnum::OpenCL(engine) => engine
                .get_main_function(context.downcast_ref().unwrap())
                .map(|f| Box::new(f) as Box<dyn Any>),
            EnginesEnum::Metal(engine) => engine
                .get_main_function(context.downcast_ref().unwrap())
                .map(|f| Box::new(f) as Box<dyn Any>),
        }
    }

    fn create_context(&self, device_index: u32) -> Result<Self::Context, anyhow::Error> {
        match self {
            EnginesEnum::Cuda(engine) => engine.create_context(device_index).map(|f| Box::new(f) as Box<dyn Any>),
            EnginesEnum::OpenCL(engine) => engine.create_context(device_index).map(|f| Box::new(f) as Box<dyn Any>),
            EnginesEnum::Metal(engine) => engine.create_context(device_index).map(|f| Box::new(f) as Box<dyn Any>),
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
        match self {
            EnginesEnum::Cuda(engine) => engine.mine(
                function.downcast_ref().unwrap(),
                context.downcast_ref().unwrap(),
                data,
                min_difficulty,
                nonce_start,
                num_iterations,
                block_size,
                grid_size,
            ),
            EnginesEnum::OpenCL(engine) => engine.mine(
                function.downcast_ref().unwrap(),
                context.downcast_ref().unwrap(),
                data,
                min_difficulty,
                nonce_start,
                num_iterations,
                block_size,
                grid_size,
            ),
            EnginesEnum::Metal(engine) => engine.mine(
                function.downcast_ref().unwrap(),
                context.downcast_ref().unwrap(),
                data,
                min_difficulty,
                nonce_start,
                num_iterations,
                block_size,
                grid_size,
            ),
        }
    }
}

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

pub struct EnginesManager {
    available_engines: Vec<EngineType>,
    selected_engine: Option<EngineType>,
}

impl EnginesManager {
    pub fn new() -> Self {
        Self {
            available_engines: Vec::new(),
            selected_engine: None,
        }
    }

    fn initialize_cuda_engine(&mut self) {
        #[cfg(feature = "nvidia")]
        self.available_engines.push(EngineType::Cuda);
    }

    fn initialize_opencl_engine(&mut self) {
        #[cfg(feature = "opencl")]
        self.available_engines.push(EngineType::OpenCL);
    }

    fn initialize_metal_engine(&mut self) {
        #[cfg(feature = "metal")]
        self.available_engines.push(EngineType::Metal);
    }

    pub fn initialize_engines(&mut self) {
        self.initialize_cuda_engine();
        self.initialize_opencl_engine();
        self.initialize_metal_engine();
    }

    pub fn select_engine(&mut self, engine: EngineType) {
        self.selected_engine = Some(engine);
    }

    pub fn get_selected_engine_implementation(&self) -> Result<impl EngineImpl, anyhow::Error> {
        match self.selected_engine {
            Some(EngineType::Cuda) => Ok(EnginesEnum::Cuda(GpuEngine::new(CudaEngine::new()))),
            Some(EngineType::OpenCL) => Ok(EnginesEnum::OpenCL(GpuEngine::new(OpenClEngine::new()))),
            Some(EngineType::Metal) => Ok(EnginesEnum::Metal(GpuEngine::new(OpenClEngine::new()))),
            _ => return Err(anyhow!("Engine not selected")),
        }
    }

    pub fn get_engine_implementation_for_all_available_engines(&self) -> Result<Vec<impl EngineImpl>, anyhow::Error> {
        self.available_engines
            .iter()
            .map(|engine| match engine {
                EngineType::Cuda => Ok(EnginesEnum::Cuda(GpuEngine::new(CudaEngine::new()))),
                EngineType::OpenCL => Ok(EnginesEnum::OpenCL(GpuEngine::new(OpenClEngine::new()))),
                EngineType::Metal => Ok(EnginesEnum::Metal(GpuEngine::new(OpenClEngine::new()))),
                _ => Err(anyhow!("Engine not available")),
            })
            .collect()
    }
}

use metal::Device;

use crate::{context_impl::ContextImpl, engine_impl::EngineImpl, function_impl::FunctionImpl};

pub struct MetalContext {}
impl ContextImpl for MetalContext {

}

pub struct MetalFunction {}
impl FunctionImpl for MetalFunction {
    type Device = usize;

    fn suggested_launch_configuration(&self, device_index: &usize) -> Result<(u32, u32), anyhow::Error> {
        todo!()
    }
}
pub struct MetalEngine {}

impl EngineImpl for MetalEngine {
    type Context = MetalContext;
    type Function = MetalFunction;

    fn num_devices(&self) -> Result<u32, anyhow::Error> {
        Ok(Device::all().len() as u32)
    }

    fn create_context(&self, device_index: u32) -> Result<Self::Context, anyhow::Error> {
        todo!()
    }

    fn create_main_function(&self, context: &Self::Context) -> Result<Self::Function, anyhow::Error> {
        todo!()
    }

    fn detect_devices(&self) -> Result<Vec<crate::gpu_status_file::GpuStatus>, anyhow::Error> {
        todo!()
    }

    fn init(&mut self) -> Result<(), anyhow::Error> {
        todo!()
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
        
        todo!()
    }
}
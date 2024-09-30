use crate::{context_impl::ContextImpl, function_impl::FunctionImpl, gpu_status_file::GpuStatus};

pub trait EngineImpl {
    type Context: ContextImpl;
    type Function: FunctionImpl;
    fn init(&mut self) -> Result<(), anyhow::Error>;

    fn num_devices(&self) -> Result<u32, anyhow::Error>;

    fn detect_devices(&self) -> Result<Vec<GpuStatus>, anyhow::Error>;

    fn create_context(&self, device_index: u32) -> Result<Self::Context, anyhow::Error>;

    fn create_main_function(&self, context: &Self::Context) -> Result<Self::Function, anyhow::Error>;

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
    ) -> Result<(Option<u64>, u32, u64), anyhow::Error>;
}

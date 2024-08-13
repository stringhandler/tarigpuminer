use crate::context_impl::ContextImpl;

use crate::function_impl::FunctionImpl;

pub trait EngineImpl {
    type Context: ContextImpl;
    type Function: FunctionImpl;
    fn init(&self) -> Result<(), anyhow::Error>;

    fn num_devices(&self) -> Result<u32, anyhow::Error>;

    fn create_context(&self) -> Result<Self::Context, anyhow::Error>;

    fn create_main_function(&self) -> Result<Self::Function, anyhow::Error>;

    fn mine(&self) -> Result<(Option<u64>, u32, i32), anyhow::Error>;
}
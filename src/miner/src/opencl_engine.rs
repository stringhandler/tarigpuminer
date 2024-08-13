use anyhow::Error;
use crate::context_impl::ContextImpl;
use crate::engine_impl::EngineImpl;
use crate::function_impl::FunctionImpl;

pub struct OpenClEngine {

}

impl OpenClEngine {
    pub fn new() -> Self {
        OpenClEngine {}
    }
}

impl EngineImpl for OpenClEngine {
    type Context: OpenClContext;
    type Function: OpenClFunction;
    fn init(&self) -> Result<(), anyhow::Error>;

    fn num_devices(&self) -> Result<u32, anyhow::Error>;

    fn create_context(&self) -> Result<Self::Context, anyhow::Error>;

    fn create_main_function(&self) -> Result<Self::Function, anyhow::Error>;

    fn mine(&self) -> Result<(), Error> {
        todo!()
    }

}

pub struct OpenClContext{

}

impl ContextImpl for OpenClContext {
    fn create(&self) -> Result<Self, anyhow::Error> {
        todo!()
    }
}

pub struct OpenClFunction {}
impl FunctionImpl for OpenClFunction{
    fn suggested_launch_configuration(&self) -> (u32, u32) {
        todo!()
    }
}
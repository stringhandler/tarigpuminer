use anyhow::Error;
use crate::context_impl::ContextImpl;
use crate::engine_impl::EngineImpl;
use crate::function_impl::FunctionImpl;

#[derive(Clone)]
pub struct OpenClEngine {

}

impl OpenClEngine {
    pub fn new() -> Self {
        OpenClEngine {}
    }
}

impl EngineImpl for OpenClEngine {
    type Context = OpenClContext;
    type Function=  OpenClFunction;
    fn init(&self) -> Result<(), anyhow::Error> {
        todo!()
    }

    fn num_devices(&self) -> Result<u32, anyhow::Error>{
        todo!()
    }

    fn create_context(&self) -> Result<Self::Context, anyhow::Error> {
        todo!()
    }

    fn create_main_function(&self) -> Result<Self::Function, anyhow::Error>{
        todo!()
    }

    fn mine(&self) -> Result<(Option<u64>, u32, i32), Error> {
        todo!()
    }

}

pub struct OpenClContext{

}

impl ContextImpl for OpenClContext {

}

pub struct OpenClFunction {}
impl FunctionImpl for OpenClFunction{
    fn suggested_launch_configuration(&self) -> Result<(u32, u32), anyhow::Error> {
        todo!()
    }
}
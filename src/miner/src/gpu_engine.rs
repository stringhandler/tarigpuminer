

use crate::engine_impl::EngineImpl;





#[derive(Clone)]
pub struct GpuEngine<TEngineImpl: EngineImpl> {
    inner: TEngineImpl
}

impl<TEngineImpl: EngineImpl> GpuEngine<TEngineImpl>{

    pub fn new(engine: TEngineImpl) -> Self {
        GpuEngine {
            inner: engine
        }
    }
    pub fn init(&self) -> Result<(), anyhow::Error>{
        self.inner.init()
    }

    pub fn num_devices(&self) -> Result<u32, anyhow::Error> {
        self.inner.num_devices()
    }

    pub fn create_context(&self) -> Result<TEngineImpl::Context, anyhow::Error> {
        self.inner.create_context()
    }

    pub fn get_main_function(&self) -> Result<TEngineImpl::Function, anyhow::Error> {
        self.inner.create_main_function()
        // match self {
        //     GpuEngine::Cuda => {
        //         let module = Module::from_ptx(include_str!("../cuda/keccak.ptx"), &[
        //             ModuleJitOption::GenerateLineInfo(true),
        //         ])
        //             .context("module bad")?;
        //
        //         let func = module.get_function("keccakKernel").context("module getfunc")?;
        //         todo!()
        //     },
        //     GpuEngine::OpenCL => {
        //         todo!()
        //     }
        // }
    }

    pub fn mine(&self) -> Result<(Option<u64>, u32, i32), anyhow::Error> {
        self.inner.mine()
    }
}
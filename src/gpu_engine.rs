use crate::{engine_impl::EngineImpl, gpu_status_file::GpuStatus};

#[derive(Clone)]
pub struct GpuEngine<TEngineImpl: EngineImpl> {
    inner: TEngineImpl,
}

impl<TEngineImpl: EngineImpl> GpuEngine<TEngineImpl> {
    pub fn new(engine: TEngineImpl) -> Self {
        GpuEngine { inner: engine }
    }

    pub fn init(&mut self) -> Result<(), anyhow::Error> {
        self.inner.init()
    }

    pub fn num_devices(&self) -> Result<u32, anyhow::Error> {
        self.inner.num_devices()
    }

    pub fn detect_devices(&self) -> Result<Vec<GpuStatus>, anyhow::Error> {
        self.inner.detect_devices()
    }

    pub fn create_context(&self, device_index: u32) -> Result<TEngineImpl::Context, anyhow::Error> {
        self.inner.create_context(device_index)
    }

    pub fn get_main_function(&self, context: &TEngineImpl::Context) -> Result<TEngineImpl::Function, anyhow::Error> {
        self.inner.create_main_function(context)
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

    pub fn mine(
        &self,
        function: &TEngineImpl::Function,
        context: &TEngineImpl::Context,
        data: &[u64],
        min_difficulty: u64,
        nonce_start: u64,
        num_iterations: u32,
        block_size: u32,
        grid_size: u32,
    ) -> Result<(Option<u64>, u32, u64), anyhow::Error> {
        self.inner.mine(
            function,
            context,
            data,
            min_difficulty,
            nonce_start,
            num_iterations,
            block_size,
            grid_size,
        )
    }
}

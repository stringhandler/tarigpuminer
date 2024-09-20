use std::time::Instant;

use anyhow::Error;
#[cfg(feature = "nvidia")]
use cust::{
    device::DeviceAttribute,
    memory::{AsyncCopyDestination, DeviceCopy},
    module::{ModuleJitOption, ModuleJitOption::DetermineTargetFromContext},
    prelude::{Module, *},
};
use log::{error, info, warn};

use crate::{context_impl::ContextImpl, EngineImpl, FunctionImpl};
const LOG_TARGET: &str = "tari::gpuminer::cuda";
#[derive(Clone)]
pub struct CudaEngine {}

impl CudaEngine {
    pub fn new() -> Self {
        Self {}
    }
}

impl EngineImpl for CudaEngine {
    type Context = CudaContext;
    type Function = CudaFunction;

    fn init(&mut self) -> Result<(), anyhow::Error> {
        info!(target: LOG_TARGET, "Init CUDA Engine");
        cust::init(CudaFlags::empty())?;
        Ok(())
    }

    fn num_devices(&self) -> Result<u32, anyhow::Error> {
        let num_devices = Device::num_devices()?;
        Ok(num_devices)
    }

    fn create_context(&self, device_index: u32) -> Result<Self::Context, anyhow::Error> {
        let context = Context::new(Device::get_device(device_index)?)?;
        context.set_flags(ContextFlags::SCHED_YIELD)?;

        Ok(CudaContext { context })
    }

    fn create_main_function(&self, context: &Self::Context) -> Result<Self::Function, anyhow::Error> {
        info!(target: LOG_TARGET, "Create CUDA main function");
        let module = Module::from_ptx(include_str!("../cuda/keccak.ptx"), &[
            ModuleJitOption::GenerateLineInfo(true),
        ])?;
        // let func = context.module.get_function("keccakKernel")?;
        Ok(CudaFunction { module })
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
    ) -> Result<(Option<u64>, u32, u64), Error> {
        info!(target: LOG_TARGET, "CUDA: start mining");
        let output = vec![0u64; 5];
        let mut output_buf = output.as_slice().as_dbuf()?;

        let mut data_buf = data.as_dbuf()?;
        data_buf.copy_from(data).expect("Could not copy data to buffer");
        output_buf.copy_from(&output).expect("Could not copy output to buffer");

        let num_streams = 1;
        let mut streams = Vec::with_capacity(num_streams);
        let func = function.module.get_function("keccakKernel")?;

        let output = vec![0u64; 5];

        for st in 0..num_streams {
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

            streams.push(stream);
        }

        let data_ptr = data_buf.as_device_ptr();
        for st in 0..num_streams {
            let stream = &streams[st];
            unsafe {
                launch!(
                    func<<<grid_size, block_size, 0, stream>>>(
                    data_ptr,
                         nonce_start,
                         min_difficulty,
                         num_iterations,
                         output_buf.as_device_ptr(),

                    )
                )?;
            }
        }

        for st in 0..num_streams {
            let mut out1 = vec![0u64; 5];

            unsafe {
                output_buf.copy_to(&mut out1)?;
            }
            // stream.synchronize()?;

            if out1[0] > 0 {
                return Ok((Some((&out1[0]).clone()), grid_size * block_size * num_iterations, 0));
            }
        }

        Ok((None, grid_size * block_size * num_iterations, 0))
    }
}

pub struct CudaContext {
    context: Context,
}

impl CudaContext {}

impl ContextImpl for CudaContext {}

pub struct CudaFunction {
    module: Module,
}
impl FunctionImpl for CudaFunction {
    fn suggested_launch_configuration(&self) -> Result<(u32, u32), anyhow::Error> {
        let func = self.module.get_function("keccakKernel")?;
        let (grid_size, block_size) = func.suggested_launch_configuration(0, 0.into())?;
        // Ok((grid_size, block_size))
        Ok((1000, 100))
    }
}

use std::time::Instant;

use anyhow::Error;
#[cfg(feature = "nvidia")]
use cust::{
    device::DeviceAttribute,
    memory::{AsyncCopyDestination, DeviceCopy},
    module::{ModuleJitOption, ModuleJitOption::DetermineTargetFromContext},
    prelude::{Module, *},
};
use log::{debug, error, info, warn};

use crate::{
    context_impl::ContextImpl,
    engine_impl::EngineImpl,
    function_impl::FunctionImpl,
    gpu_status_file::GpuStatus,
    multi_engine_wrapper::EngineType,
};
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

    fn get_engine_type(&self) -> EngineType {
        EngineType::Cuda
    }

    fn detect_devices(&self) -> Result<Vec<GpuStatus>, anyhow::Error> {
        info!(target: LOG_TARGET, "Detect CUDA devices");
        let num_devices = Device::num_devices()?;
        let mut total_devices = 0;
        let mut devices = Vec::with_capacity(num_devices as usize);
        for i in 0..num_devices {
            let device = Device::get_device(i)?;
            let name = device.name()?;
            let mut gpu = GpuStatus {
                device_name: name.clone(),
                recommended_block_size: 0,
                device_index: i,
                recommended_grid_size: 0,
                max_grid_size: device.get_attribute(DeviceAttribute::MaxGridDimX).unwrap_or_default() as u32,
            };
            if let Ok(context) = self
                .create_context(u32::try_from(i).unwrap())
                .inspect_err(|e| error!(target: LOG_TARGET, "Could not create context {:?}", e))
            {
                if let Ok(func) = self
                    .create_main_function(&context)
                    .inspect_err(|e| error!(target: LOG_TARGET, "Could not create function {:?}", e))
                {
                    if let Ok((grid, block)) = func.suggested_launch_configuration(&(i as usize)) {
                        gpu.recommended_grid_size = grid;
                        gpu.recommended_block_size = block;
                    }
                    devices.push(gpu);
                    total_devices += 1;
                    debug!(target: LOG_TARGET, "Device nr {:?}: {}", total_devices, name);
                    println!("Device nr {:?}: {}", total_devices, name);
                }
            }
        }
        if devices.len() > 0 {
            return Ok(devices);
        }
        return Err(anyhow::anyhow!("No gpu device detected"));
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
        // println!("CUDA: start mining");
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
    type Device = usize;

    fn suggested_launch_configuration(&self, device: &Self::Device) -> Result<(u32, u32), anyhow::Error> {
        let func = self.module.get_function("keccakKernel")?;
        let (grid_size, block_size) = func.suggested_launch_configuration(*device, 0.into())?;
        Ok((grid_size, block_size))
    }
}

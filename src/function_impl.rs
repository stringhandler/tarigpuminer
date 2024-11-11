pub trait FunctionImpl {
    type Device;
    fn suggested_launch_configuration(&self, device: &Self::Device) -> Result<(u32, u32), anyhow::Error>;
}

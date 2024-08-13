pub trait FunctionImpl {

    fn suggested_launch_configuration(&self) -> Result<(u32, u32), anyhow::Error>;
}


use minotari_app_grpc::tari_rpc::{
    base_node_client::BaseNodeClient,
    pow_algo::PowAlgos,
    AggregateBody,
    Block,
    Empty,
    GetNewBlockResult,
    NewBlockTemplate,
    NewBlockTemplateRequest,
    NewBlockTemplateResponse,
    PowAlgo,
};
use tari_core::validation::aggregate_body;
use tonic::async_trait;

use crate::Cli;

pub(crate) struct BaseNodeClientWrapper {
    client: BaseNodeClient<tonic::transport::Channel>,
}

impl BaseNodeClientWrapper {
    pub async fn connect(url: &str) -> Result<Self, anyhow::Error> {
        println!("Connecting to {}", url);
        let client = BaseNodeClient::connect(url.to_string()).await?;
        Ok(Self { client })
    }
}

#[async_trait]
impl NodeClient for BaseNodeClientWrapper {
    async fn get_version(&mut self) -> Result<u64, anyhow::Error> {
        let res = self.client.get_version(tonic::Request::new(Empty {})).await?;
        // dbg!(res);
        Ok(0)
    }

    async fn get_block_template(&mut self) -> Result<NewBlockTemplateResponse, anyhow::Error> {
        let res = self
            .client
            .get_new_block_template(tonic::Request::new({
                NewBlockTemplateRequest {
                    max_weight: 0,
                    algo: Some(PowAlgo {
                        pow_algo: PowAlgos::Sha3x.into(),
                    }),
                }
            }))
            .await?;
        Ok(res.into_inner())
    }

    async fn get_new_block(&mut self, template: NewBlockTemplate) -> Result<GetNewBlockResult, anyhow::Error> {
        let res = self.client.get_new_block(tonic::Request::new(template)).await?;
        Ok(res.into_inner())
    }

    async fn submit_block(&mut self, block: Block) -> Result<(), anyhow::Error> {
        // dbg!(&block);
        let res = self.client.submit_block(tonic::Request::new(block)).await?;
        println!("Block submitted: {:?}", res);
        Ok(())
    }
}

#[async_trait]
pub trait NodeClient {
    async fn get_version(&mut self) -> Result<u64, anyhow::Error>;

    async fn get_block_template(&mut self) -> Result<NewBlockTemplateResponse, anyhow::Error>;

    async fn get_new_block(&mut self, template: NewBlockTemplate) -> Result<GetNewBlockResult, anyhow::Error>;

    async fn submit_block(&mut self, block: Block) -> Result<(), anyhow::Error>;
}

pub(crate) async fn create_client(url: &str, benchmark: bool) -> Result<Client, anyhow::Error> {
    if benchmark {
        return Ok(Client::Benchmark(BenchmarkNodeClient {}));
    }
    Ok(Client::BaseNode(BaseNodeClientWrapper::connect(url).await?))
}

pub(crate) enum Client {
    BaseNode(BaseNodeClientWrapper),
    Benchmark(BenchmarkNodeClient),
}

impl Client {
    pub async fn get_version(&mut self) -> Result<u64, anyhow::Error> {
        match self {
            Client::BaseNode(client) => client.get_version().await,
            Client::Benchmark(client) => client.get_version().await,
        }
    }

    pub async fn get_block_template(&mut self) -> Result<NewBlockTemplateResponse, anyhow::Error> {
        match self {
            Client::BaseNode(client) => client.get_block_template().await,
            Client::Benchmark(client) => client.get_block_template().await,
        }
    }

    pub async fn get_new_block(&mut self, template: NewBlockTemplate) -> Result<GetNewBlockResult, anyhow::Error> {
        match self {
            Client::BaseNode(client) => client.get_new_block(template).await,
            Client::Benchmark(client) => client.get_new_block(template).await,
        }
    }

    pub async fn submit_block(&mut self, block: Block) -> Result<(), anyhow::Error> {
        match self {
            Client::BaseNode(client) => client.submit_block(block).await,
            Client::Benchmark(client) => client.submit_block(block).await,
        }
    }
}

pub(crate) struct BenchmarkNodeClient {}

#[async_trait]
impl NodeClient for BenchmarkNodeClient {
    async fn get_version(&mut self) -> Result<u64, anyhow::Error> {
        Ok(0)
    }

    async fn get_block_template(&mut self) -> Result<NewBlockTemplateResponse, anyhow::Error> {
        todo!()
    }

    async fn get_new_block(&mut self, template: NewBlockTemplate) -> Result<GetNewBlockResult, anyhow::Error> {
        todo!()
    }

    async fn submit_block(&mut self, block: Block) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

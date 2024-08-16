use anyhow::{anyhow, Error};
use minotari_app_grpc::authentication::ClientAuthenticationInterceptor;
use minotari_app_grpc::tari_rpc::{Block, GetNewBlockRequest, GetNewBlockResult, NewBlockTemplate, NewBlockTemplateResponse, SubmitBlockRequest};
use minotari_app_grpc::tari_rpc::base_node_client::BaseNodeClient;
use minotari_app_grpc::tari_rpc::sha_p2_pool_client::ShaP2PoolClient;
use tari_common_types::tari_address::TariAddress;
use tonic::async_trait;
use tonic::codegen::InterceptedService;
use tonic::transport::Channel;
use crate::node_client::{NodeClient};

pub struct P2poolClientWrapper {
    client: ShaP2PoolClient<Channel>,
    wallet_payment_address: TariAddress,
}

impl P2poolClientWrapper {
    pub async fn connect(url: &str, wallet_payment_address: TariAddress) -> Result<Self, anyhow::Error> {
        println!("Connecting to {}", url);
        let client = ShaP2PoolClient::connect(url.to_string()).await?;
        Ok(Self { client, wallet_payment_address })
    }
}

#[async_trait]
impl NodeClient for P2poolClientWrapper {
    async fn get_version(&mut self) -> Result<u64, Error> {
        Ok(0)
    }

    async fn get_block_template(&mut self) -> Result<NewBlockTemplateResponse, Error> {
        Err(anyhow!("not supported"))
    }

    async fn get_new_block(&mut self, _template: NewBlockTemplate) -> Result<GetNewBlockResult, Error> {
        let result = self.client.get_new_block(GetNewBlockRequest::default()).await?.into_inner();
        if let Some(block_result) = result.block {
            return Ok(block_result);
        }
        
        Err(anyhow!("no block in response"))
    }

    async fn submit_block(&mut self, block: Block) -> Result<(), Error> {
        self.client.submit_block(SubmitBlockRequest{
            block: Some(block), 
            wallet_payment_address: self.wallet_payment_address.to_base58(),
        })
    }
}
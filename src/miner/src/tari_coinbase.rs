use rand::rngs::OsRng;
use tari_common_types::tari_address::TariAddress;
use tari_common_types::types::PublicKey;
use tari_core::consensus::ConsensusConstants;
use tari_core::one_sided::diffie_hellman_stealth_domain_hasher;
use tari_core::one_sided::shared_secret_to_output_encryption_key;
use tari_core::one_sided::shared_secret_to_output_spending_key;
use tari_core::one_sided::stealth_address_script_spending_key;
use tari_core::transactions::key_manager::TransactionKeyManagerBranch;
use tari_core::transactions::key_manager::TransactionKeyManagerInterface;
use tari_core::transactions::key_manager::{MemoryDbKeyManager, TariKeyId};
use tari_core::transactions::tari_amount::MicroMinotari;
use tari_core::transactions::transaction_components::WalletOutput;
use tari_core::transactions::transaction_components::{
    RangeProofType, Transaction, TransactionKernel, TransactionOutput,
};
use tari_core::transactions::CoinbaseBuildError;
use tari_core::transactions::CoinbaseBuilder;
use tari_crypto::keys::PublicKey as PK;
use tari_key_manager::key_manager_service::KeyManagerInterface;
use tari_script::one_sided_payment_script;
use tari_script::stealth_payment_script;

pub async fn generate_coinbase(
    fee: MicroMinotari,
    reward: MicroMinotari,
    height: u64,
    extra: &[u8],
    key_manager: &MemoryDbKeyManager,
    wallet_payment_address: &TariAddress,
    stealth_payment: bool,
    consensus_constants: &ConsensusConstants,
    range_proof_type: RangeProofType,
) -> Result<(TransactionOutput, TransactionKernel), CoinbaseBuildError> {
    let script_key_id = TariKeyId::default();
    let (_, coinbase_output, coinbase_kernel, _) = generate_coinbase_with_wallet_output(
        fee,
        reward,
        height,
        extra,
        key_manager,
        &script_key_id,
        wallet_payment_address,
        stealth_payment,
        consensus_constants,
        range_proof_type,
    )
    .await?;
    Ok((coinbase_output, coinbase_kernel))
}

pub async fn generate_coinbase_with_wallet_output(
    fee: MicroMinotari,
    reward: MicroMinotari,
    height: u64,
    extra: &[u8],
    key_manager: &MemoryDbKeyManager,
    script_key_id: &TariKeyId,
    wallet_payment_address: &TariAddress,
    stealth_payment: bool,
    consensus_constants: &ConsensusConstants,
    range_proof_type: RangeProofType,
) -> Result<
    (
        Transaction,
        TransactionOutput,
        TransactionKernel,
        WalletOutput,
    ),
    CoinbaseBuildError,
> {
    let (sender_offset_key_id, _) = key_manager
        .get_next_key(TransactionKeyManagerBranch::SenderOffset.get_branch_key())
        .await?;
    let shared_secret = key_manager
        .get_diffie_hellman_shared_secret(
            &sender_offset_key_id,
            wallet_payment_address.public_key(),
        )
        .await?;
    let spending_key = shared_secret_to_output_spending_key(&shared_secret)?;

    let encryption_private_key = shared_secret_to_output_encryption_key(&shared_secret)?;
    let encryption_key_id = key_manager.import_key(encryption_private_key).await?;

    let spending_key_id = key_manager.import_key(spending_key).await?;

    let script = if stealth_payment {
        let (nonce_private_key, nonce_public_key) = PublicKey::random_keypair(&mut OsRng);
        let c = diffie_hellman_stealth_domain_hasher(
            &nonce_private_key,
            wallet_payment_address.public_key(),
        );
        let script_spending_key =
            stealth_address_script_spending_key(&c, wallet_payment_address.public_key());
        stealth_payment_script(&nonce_public_key, &script_spending_key)
    } else {
        one_sided_payment_script(wallet_payment_address.public_key())
    };

    let (transaction, wallet_output) = CoinbaseBuilder::new(key_manager.clone())
        .with_block_height(height)
        .with_fees(fee)
        .with_spend_key_id(spending_key_id)
        .with_encryption_key_id(encryption_key_id)
        .with_sender_offset_key_id(sender_offset_key_id)
        .with_script_key_id(script_key_id.clone())
        .with_script(script)
        .with_extra(extra.to_vec())
        .with_range_proof_type(range_proof_type)
        .build_with_reward(consensus_constants, reward)
        .await?;

    let output = transaction
        .body()
        .outputs()
        .first()
        .ok_or(CoinbaseBuildError::BuildError(
            "No output found".to_string(),
        ))?;
    let kernel = transaction
        .body()
        .kernels()
        .first()
        .ok_or(CoinbaseBuildError::BuildError(
            "No kernel found".to_string(),
        ))?;

    Ok((
        transaction.clone(),
        output.clone(),
        kernel.clone(),
        wallet_output,
    ))
}

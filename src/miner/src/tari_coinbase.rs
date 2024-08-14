use rand::rngs::OsRng;
use tari_common_types::{tari_address::TariAddress, types::PublicKey};
use tari_core::{
    consensus::ConsensusConstants,
    one_sided::{
        diffie_hellman_stealth_domain_hasher,
        shared_secret_to_output_encryption_key,
        shared_secret_to_output_spending_key,
    },
    transactions::{
        key_manager::{MemoryDbKeyManager, TariKeyId, TransactionKeyManagerBranch, TransactionKeyManagerInterface},
        tari_amount::MicroMinotari,
        transaction_components::{RangeProofType, Transaction, TransactionKernel, TransactionOutput, WalletOutput},
        CoinbaseBuildError,
        CoinbaseBuilder,
    },
};
use tari_crypto::keys::PublicKey as PK;
use tari_key_manager::key_manager_service::KeyManagerInterface;

use tari_core::transactions::transaction_components::encrypted_data::PaymentId;

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
    let (_, coinbase_output, coinbase_kernel, _) = tari_core::transactions::generate_coinbase_with_wallet_output(
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
        PaymentId::Empty
    )
    .await?;
    Ok((coinbase_output, coinbase_kernel))
}

// pub async fn generate_coinbase_with_wallet_output(
//     fee: MicroMinotari,
//     reward: MicroMinotari,
//     height: u64,
//     extra: &[u8],
//     key_manager: &MemoryDbKeyManager,
//     script_key_id: &TariKeyId,
//     wallet_payment_address: &TariAddress,
//     stealth_payment: bool,
//     consensus_constants: &ConsensusConstants,
//     range_proof_type: RangeProofType,
// ) -> Result<(Transaction, TransactionOutput, TransactionKernel, WalletOutput), CoinbaseBuildError> {
//     let (sender_offset_key_id, _) = key_manager
//         .get_next_key(TransactionKeyManagerBranch::SenderOffset.get_branch_key())
//         .await?;
//     let shared_secret = key_manager
//         .get_diffie_hellman_shared_secret(&sender_offset_key_id, wallet_payment_address.public_key())
//         .await?;
//     let spending_key = shared_secret_to_output_spending_key(&shared_secret)?;

//     let encryption_private_key = shared_secret_to_output_encryption_key(&shared_secret)?;
//     let encryption_key_id = key_manager.import_key(encryption_private_key).await?;

//     let spending_key_id = key_manager.import_key(spending_key).await?;

//     let script = if stealth_payment {
//         let (nonce_private_key, nonce_public_key) = PublicKey::random_keypair(&mut OsRng);
//         let c = diffie_hellman_stealth_domain_hasher(&nonce_private_key, wallet_payment_address.public_key());
//         let script_spending_key = stealth_address_script_spending_key(&c, wallet_payment_address.public_key());
//         stealth_payment_script(&nonce_public_key, &script_spending_key)
//     } else {
//         one_sided_payment_script(wallet_payment_address.public_key())
//     };

//     let (transaction, wallet_output) = CoinbaseBuilder::new(key_manager.clone())
//         .with_block_height(height)
//         .with_fees(fee)
//         .with_spend_key_id(spending_key_id)
//         .with_encryption_key_id(encryption_key_id)
//         .with_sender_offset_key_id(sender_offset_key_id)
//         .with_script_key_id(script_key_id.clone())
//         .with_script(script)
//         .with_extra(extra.to_vec())
//         .with_range_proof_type(range_proof_type)
//         .build_with_reward(consensus_constants, reward)
//         .await?;

//     let output = transaction
//         .body()
//         .outputs()
//         .first()
//         .ok_or(CoinbaseBuildError::BuildError("No output found".to_string()))?;
//     let kernel = transaction
//         .body()
//         .kernels()
//         .first()
//         .ok_or(CoinbaseBuildError::BuildError("No kernel found".to_string()))?;

//     Ok((transaction.clone(), output.clone(), kernel.clone(), wallet_output))
// }

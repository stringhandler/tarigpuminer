use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::http::server::AppState;

#[derive(Serialize, Deserialize)]
pub struct Stats {
    pub hashes_per_second: u64,
    pub accepted_blocks: u64,
    pub rejected_blocks: u64,
}

pub async fn handle_get_stats(State(state): State<AppState>) -> Result<Json<Stats>, StatusCode> {
    Ok(Json(Stats {
        hashes_per_second: state.stats_store.hashes_per_second(),
        accepted_blocks: state.stats_store.accepted_blocks(),
        rejected_blocks: state.stats_store.rejected_blocks(),
    }))
}

use std::collections::HashMap;

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::http::{
    server::AppState,
    stats_collector::{AverageHashrate, GetHashrateResponse},
};

#[derive(Serialize)]
pub struct Stats {
    hashrate_per_device: HashMap<usize, AverageHashrate>,
    total_hashrate: AverageHashrate,
}

pub async fn handle_get_stats(State(state): State<AppState>) -> Result<Json<Stats>, StatusCode> {
    let hashrate = state.stats_client.get_hashrate().await.map_err(|e| {
        log::error!("Failed to get hashrate: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let stats = Stats {
        hashrate_per_device: hashrate.devices,
        total_hashrate: hashrate.total,
    };
    Ok(Json(stats))
}

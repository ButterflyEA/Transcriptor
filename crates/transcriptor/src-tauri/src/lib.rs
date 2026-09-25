mod app;
mod core;
mod error;
mod state;

use std::sync::Arc;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Arc::new(state::AppState::default()))
        .invoke_handler(tauri::generate_handler![
            app::get_defaults,
            app::inspect_audio,
            app::transcribe,
            app::stop,
            app::save_transcript,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
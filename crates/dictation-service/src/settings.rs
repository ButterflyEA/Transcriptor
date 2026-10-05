//! Settings window (`dictation settings`), a separate process from the service.

use std::path::{Path, PathBuf};

use cpal::traits::{DeviceTrait, HostTrait};
use eframe::egui;

use crate::chord::Chord;
use crate::config::{self, Config, parse_model};
use crate::control::{self, Command};
use crate::engine::resolve_model_dir;

#[derive(Clone, Debug, PartialEq)]
pub struct FormState {
    pub hotkey: String,
    pub model: String,
    pub model_dir: String,
    pub language: String,
    /// Empty means the default input device.
    pub input_device: String,
    base: Config,
}

impl FormState {
    pub fn from_config(c: &Config) -> Self {
        Self {
            hotkey: c.hotkey.clone(),
            model: c.model.clone(),
            model_dir: c
                .model_dir
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            language: c.language.clone(),
            input_device: c.input_device.clone().unwrap_or_default(),
            base: c.clone(),
        }
    }
}

/// Builds a valid `Config` from the form, keeping settings the form doesn't edit.
pub fn validate_form(f: &FormState) -> Result<Config, String> {
    let mut c = f.base.clone();
    c.hotkey = f.hotkey.trim().to_string();
    c.model = f.model.trim().to_string();
    let dir = f.model_dir.trim();
    c.model_dir = (!dir.is_empty()).then(|| PathBuf::from(dir));
    c.language = f.language.trim().to_ascii_lowercase();
    let dev = f.input_device.trim();
    c.input_device = (!dev.is_empty()).then(|| dev.to_string());
    c.validate().map_err(|e| e.to_string())?;
    check_model_present(&c)?;
    Chord::parse(&c.hotkey).map_err(|e| e.to_string())?;
    Ok(c)
}

fn app_dirs() -> (PathBuf, PathBuf) {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_default();
    (
        exe_dir,
        dirs::data_dir().unwrap_or_default().join("dictation"),
    )
}

/// The service never downloads models, so a model without weights on disk
/// would only fail at load time. The bundled default with no explicit folder
/// is left to the service, which reports the searched paths itself.
fn check_model_present(c: &Config) -> Result<(), String> {
    if c.model == "ivrit-turbo" && c.model_dir.is_none() {
        return Ok(());
    }
    let (exe, data) = app_dirs();
    if resolve_model_dir(&c.model, c.model_dir.as_deref(), &exe, &data).is_some() {
        return Ok(());
    }
    Err(format!(
        "no {} weights (model.safetensors) found. Put the model in a folder and set \"Model folder\", or under {}",
        c.model,
        exe.join("models").join(&c.model).display()
    ))
}

/// Models that can be selected: the bundled one, plus any `models/<name>` folder
/// that has weights and a name the engine understands.
pub fn available_models(exe_dir: &Path, data_dir: &Path) -> Vec<String> {
    let mut out = vec!["ivrit-turbo".to_string()];
    for root in [exe_dir.join("models"), data_dir.join("models")] {
        let Ok(rd) = std::fs::read_dir(root) else {
            continue;
        };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if parse_model(&name).is_some()
                && e.path().join("model.safetensors").is_file()
                && !out.contains(&name)
            {
                out.push(name);
            }
        }
    }
    out
}

/// `ctrl+shift+space` style string for a key press, or `None` for a bare modifier.
pub fn chord_string(ctrl: bool, shift: bool, alt: bool, meta: bool, key: &str) -> String {
    let mut parts = Vec::new();
    for (on, name) in [
        (ctrl, "ctrl"),
        (shift, "shift"),
        (alt, "alt"),
        (meta, "meta"),
    ] {
        if on {
            parts.push(name.to_string());
        }
    }
    parts.push(key.to_string());
    parts.join("+")
}

fn key_name(key: egui::Key) -> String {
    let s = format!("{key:?}").to_ascii_lowercase();
    match s.strip_prefix("num") {
        Some(rest) if rest.len() == 1 => rest.to_string(),
        _ => s,
    }
}

fn input_devices() -> Vec<String> {
    cpal::default_host()
        .input_devices()
        .map(|it| it.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default()
}

const LANGS: &[&str] = &["auto", "he", "en", "ar", "ru", "fr", "de", "es"];

struct App {
    form: FormState,
    devices: Vec<String>,
    models: Vec<String>,
    capturing: bool,
    message: Option<(bool, String)>,
    wayland: bool,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.capturing {
            let pressed = ctx.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => Some(chord_string(
                        modifiers.ctrl,
                        modifiers.shift,
                        modifiers.alt,
                        modifiers.mac_cmd || modifiers.command && !modifiers.ctrl,
                        &key_name(*key),
                    )),
                    _ => None,
                })
            });
            if let Some(chord) = pressed {
                self.form.hotkey = chord;
                self.capturing = false;
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Grid::new("form").num_columns(2).show(ui, |ui| {
                ui.label("Hotkey");
                if self.wayland {
                    ui.label(
                        "Bind `dictation start`/`stop` in your compositor (see docs/dictation.md)",
                    );
                } else {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.form.hotkey).desired_width(140.0),
                        );
                        let label = if self.capturing {
                            "Press keys…"
                        } else {
                            "Record new hotkey"
                        };
                        if ui.button(label).clicked() {
                            self.capturing = true;
                        }
                    });
                }
                ui.end_row();

                ui.label("Model");
                egui::ComboBox::from_id_salt("model")
                    .selected_text(self.form.model.clone())
                    .show_ui(ui, |ui| {
                        for m in &self.models {
                            ui.selectable_value(&mut self.form.model, m.clone(), m);
                        }
                    });
                ui.end_row();

                ui.label("Model folder");
                ui.add(egui::TextEdit::singleline(&mut self.form.model_dir).desired_width(320.0));
                ui.end_row();

                ui.label("Language");
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("lang")
                        .selected_text(self.form.language.clone())
                        .show_ui(ui, |ui| {
                            for l in LANGS {
                                ui.selectable_value(&mut self.form.language, l.to_string(), *l);
                            }
                        });
                    ui.add(egui::TextEdit::singleline(&mut self.form.language).desired_width(50.0));
                });
                ui.end_row();

                ui.label("Input device");
                let shown = if self.form.input_device.is_empty() {
                    "Default".to_string()
                } else {
                    self.form.input_device.clone()
                };
                egui::ComboBox::from_id_salt("dev")
                    .selected_text(shown)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.form.input_device, String::new(), "Default");
                        for d in &self.devices {
                            ui.selectable_value(&mut self.form.input_device, d.clone(), d);
                        }
                    });
                ui.end_row();
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    self.message = Some(save(&self.form));
                }
                if ui.button("Cancel").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            if let Some((ok, m)) = &self.message {
                let color = if *ok {
                    egui::Color32::from_rgb(60, 160, 60)
                } else {
                    egui::Color32::from_rgb(220, 60, 60)
                };
                ui.colored_label(color, m);
            }
        });
    }
}

/// Validates, writes the config, then asks a running service to reload it.
fn save(form: &FormState) -> (bool, String) {
    let cfg = match validate_form(form) {
        Ok(c) => c,
        Err(e) => return (false, e),
    };
    if let Err(e) = cfg.save_to(&config::default_path()) {
        return (false, e.to_string());
    }
    let note = match control::send(Command::Reload) {
        Ok(r) if r == "ok" => "Saved. The running service is reloading.".to_string(),
        Ok(r) => format!("Saved, but the service said: {r}"),
        Err(_) => "Saved. The service is not running; it will use this on next start.".into(),
    };
    let hotkey_note = if cfg.hotkey != form.base.hotkey {
        " Restart the service to apply the new hotkey."
    } else {
        ""
    };
    (true, format!("{note}{hotkey_note}"))
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::load_from(&config::default_path())?;
    let app = App {
        form: FormState::from_config(&cfg),
        devices: input_devices(),
        models: {
            let (exe, data) = app_dirs();
            let mut m = available_models(&exe, &data);
            if !m.contains(&cfg.model) {
                m.push(cfg.model.clone());
            }
            m
        },
        capturing: false,
        message: None,
        wayland: cfg!(target_os = "linux") && crate::output::is_wayland(),
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([520.0, 280.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Dictation settings",
        options,
        Box::new(|_| Ok(Box::new(app))),
    )
    .map_err(|e| e.to_string().into())
}

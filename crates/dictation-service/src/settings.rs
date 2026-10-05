//! Settings window (`dictation settings`), a separate process from the service.

use std::path::PathBuf;

use cpal::traits::{DeviceTrait, HostTrait};
use eframe::egui;

use crate::chord::Chord;
use crate::config::{self, Config};
use crate::control::{self, Command};

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
    Chord::parse(&c.hotkey).map_err(|e| e.to_string())?;
    Ok(c)
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

const MODELS: &[&str] = &[
    "ivrit-turbo",
    "large-v3-turbo",
    "large-v3",
    "medium",
    "small",
    "base",
    "tiny",
];
const LANGS: &[&str] = &["auto", "he", "en", "ar", "ru", "fr", "de", "es"];

struct App {
    form: FormState,
    devices: Vec<String>,
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
                        for m in MODELS {
                            ui.selectable_value(&mut self.form.model, m.to_string(), *m);
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

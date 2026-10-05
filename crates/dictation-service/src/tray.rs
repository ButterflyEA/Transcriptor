//! Tray icon, menu and notifications. The tray is optional: if it cannot be
//! created the service keeps running headless.

use std::process::Command;
use std::sync::mpsc::Sender;

use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoop, EventLoopBuilder};
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use crate::controller::Status;
use crate::runtime::{Runtime, RuntimeMsg};

pub enum UiEvent {
    Status(Status),
    Notice(String),
    Menu(MenuEvent),
}

const SIZE: u32 = 32;

fn color(status: Status) -> [u8; 3] {
    match status {
        Status::Loading => [150, 150, 150],
        Status::Idle => [60, 180, 75],
        Status::Recording => [230, 40, 40],
        Status::Transcribing => [245, 160, 30],
        Status::Error => [140, 40, 160],
    }
}

/// A filled circle in the status colour, as RGBA.
pub fn icon_rgba(status: Status) -> Vec<u8> {
    let [r, g, b] = color(status);
    let c = (SIZE as f32 - 1.0) / 2.0;
    let mut out = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let d = ((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt();
            let alpha = if d <= c - 2.0 { 255 } else { 0 };
            out.extend_from_slice(&[r, g, b, alpha]);
        }
    }
    out
}

fn icon(status: Status) -> Icon {
    Icon::from_rgba(icon_rgba(status), SIZE, SIZE).expect("valid icon buffer")
}

pub fn tooltip(status: Status) -> String {
    format!("Dictation: {}", status.as_str())
}

pub fn event_loop() -> EventLoop<UiEvent> {
    EventLoopBuilder::<UiEvent>::with_user_event().build()
}

struct Tray {
    icon: TrayIcon,
    status_item: MenuItem,
    pause_item: CheckMenuItem,
    settings_item: MenuItem,
    quit_item: MenuItem,
}

fn build_tray() -> Result<Tray, String> {
    let status_item = MenuItem::new("Status: loading", false, None);
    let pause_item = CheckMenuItem::new("Paused", true, false, None);
    let settings_item = MenuItem::new("Settings…", true, None);
    let quit_item = MenuItem::new("Quit", true, None);
    let menu = Menu::new();
    menu.append_items(&[
        &status_item,
        &pause_item,
        &settings_item,
        &PredefinedMenuItem::separator(),
        &quit_item,
    ])
    .map_err(|e| e.to_string())?;
    let icon = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_icon(icon(Status::Loading))
        .with_tooltip(tooltip(Status::Loading))
        .build()
        .map_err(|e| e.to_string())?;
    Ok(Tray {
        icon,
        status_item,
        pause_item,
        settings_item,
        quit_item,
    })
}

pub fn notify(message: &str) {
    if let Err(e) = notify_rust::Notification::new()
        .summary("Dictation")
        .body(message)
        .show()
    {
        log::warn!("notification failed ({e}): {message}");
    }
}

fn open_settings() {
    let result =
        std::env::current_exe().and_then(|exe| Command::new(exe).arg("settings").spawn().map(drop));
    if let Err(e) = result {
        notify(&format!("could not open settings: {e}"));
    }
}

/// Runs the UI loop on the calling thread until Quit; never returns.
pub fn run(event_loop: EventLoop<UiEvent>, runtime: Runtime) -> ! {
    let proxy = event_loop.create_proxy();
    MenuEvent::set_event_handler(Some(move |e| {
        let _ = proxy.send_event(UiEvent::Menu(e));
    }));
    let tx: Sender<RuntimeMsg> = runtime.tx.clone();
    let mut runtime = Some(runtime);
    let mut tray: Option<Tray> = None;

    event_loop.run(move |event, _, flow| {
        *flow = ControlFlow::Wait;
        match event {
            Event::NewEvents(StartCause::Init) => match build_tray() {
                Ok(t) => tray = Some(t),
                Err(e) => eprintln!("tray unavailable ({e}); running headless"),
            },
            Event::UserEvent(UiEvent::Status(s)) => {
                if let Some(t) = &tray {
                    let _ = t.icon.set_icon(Some(icon(s)));
                    let _ = t.icon.set_tooltip(Some(tooltip(s)));
                    t.status_item.set_text(format!("Status: {}", s.as_str()));
                }
            }
            Event::UserEvent(UiEvent::Notice(m)) => notify(&m),
            Event::UserEvent(UiEvent::Menu(e)) => {
                let Some(t) = &tray else { return };
                if e.id == *t.pause_item.id() {
                    let _ = tx.send(RuntimeMsg::SetPaused(t.pause_item.is_checked()));
                } else if e.id == *t.settings_item.id() {
                    open_settings();
                } else if e.id == *t.quit_item.id() {
                    tray = None;
                    if let Some(rt) = runtime.take() {
                        rt.shutdown();
                    }
                    *flow = ControlFlow::Exit;
                }
            }
            _ => {}
        }
    })
}

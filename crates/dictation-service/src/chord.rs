use std::collections::BTreeSet;

/// A hotkey as a set of normalized lowercase key names, e.g. `ctrl+space`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chord {
    keys: BTreeSet<String>,
}

impl Chord {
    pub fn parse(s: &str) -> Result<Chord, String> {
        let keys: BTreeSet<String> = s
            .split('+')
            .map(|k| k.trim().to_ascii_lowercase())
            .filter(|k| !k.is_empty())
            .collect();
        if keys.is_empty() {
            return Err("hotkey is empty".into());
        }
        Ok(Chord { keys })
    }

    pub fn keys(&self) -> &BTreeSet<String> {
        &self.keys
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HotkeyEvent {
    Pressed,
    Released,
}

pub struct ChordTracker {
    chord: Chord,
    held: BTreeSet<String>,
    active: bool,
}

impl ChordTracker {
    pub fn new(chord: Chord) -> Self {
        Self {
            chord,
            held: BTreeSet::new(),
            active: false,
        }
    }

    /// Feed one key transition (`down` = pressed). Returns an event only when
    /// the chord as a whole becomes pressed or released.
    pub fn feed(&mut self, key: &str, down: bool) -> Option<HotkeyEvent> {
        if down {
            // Auto-repeat of an already-held key changes nothing.
            if !self.held.insert(key.to_owned()) {
                return None;
            }
        } else if !self.held.remove(key) {
            return None;
        }
        let all_down = self.chord.keys.iter().all(|k| self.held.contains(k));
        if all_down && !self.active {
            self.active = true;
            Some(HotkeyEvent::Pressed)
        } else if !all_down && self.active {
            self.active = false;
            Some(HotkeyEvent::Released)
        } else {
            None
        }
    }
}

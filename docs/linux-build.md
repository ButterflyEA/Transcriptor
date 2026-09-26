# Building on Linux

Tauri reads `crates/transcriptor/src-tauri/tauri.linux.conf.json` automatically
when building on Linux and merges it over `tauri.conf.json`. The Windows
config is untouched, so `tauri build` on Windows still produces only NSIS and
MSI, and on Linux it produces AppImage, deb and rpm.

## Read this before you build: glibc

**Building the deb and rpm on Arch produces packages tied to Arch's glibc, and
those will not install on older Debian, Ubuntu or Fedora systems.** This is not
a Tauri quirk, it is how dynamically linked binaries work, and the Tauri docs
call it out directly: you must build on the *oldest* system you intend to
support. Arch is a rolling release and is almost always the newest glibc in
common use, so it is close to the worst possible build host for portable
packages.

What that means in practice:

| Artefact | Built on Arch | Who it actually works for |
|---|---|---|
| AppImage | fine | Arch and other up-to-date systems — this is your own platform, so it's the easy case |
| deb | glibc too new | Recent/newer Debian and Ubuntu only; will fail on Debian 12 / Ubuntu 22.04 with `GLIBC_2.xx not found` |
| rpm | glibc too new | Recent Fedora only |

So decide up front which you actually want:

- **Just testing on your own machine?** Build all three on Arch. The AppImage
  is the one you'll use; the deb and rpm are throwaway.
- **Shipping the deb and rpm to real users?** Build them in a Debian 12
  container, not on Arch. Debian 12 is the documented baseline because it ships
  `libwebkit2gtk-4.1` from its own repositories. A container also keeps the
  host clean.

## Prerequisites (Arch / Omarchy)

```sh
sudo pacman -S --needed \
  webkit2gtk-4.1 base-devel curl wget file openssl librsvg \
  patchelf libfuse2

# bundler tooling - only for the targets you actually want
sudo pacman -S --needed dpkg   # provides dpkg-deb, for the .deb
sudo pacman -S --needed rpm    # provides rpmbuild, for the .rpm
```

`libfuse2` is not needed to *build* the AppImage, only to *run* it on Arch
(current Arch ships `fuse3`, and AppImages still want `fuse2`). Without it the
AppImage will refuse to start; `--appimage-extract-and-run` is the escape
hatch if you'd rather not install it.

The AppImage target downloads its own tooling at build time, so it needs
network access on first run.

If you skip `dpkg` or `rpm` and still list those targets, the build fails at
the bundling stage — after the full Rust compile, so it's a slow way to find
out. Build one target at a time while you get it working:

```sh
npm run tauri build -- --bundles appimage
```

## Build

```sh
cd crates/transcriptor/ui
npm install          # first time only
npm run tauri build
```

All three targets at once, or one at a time with `--bundles deb`, `--bundles
rpm`, `--bundles appimage`.

## Where the output lands

`target/release/bundle/` at the **repo root** (the workspace target dir, shared
with the Rust crates):

```
target/release/bundle/appimage/Transcriptor_1.0.0_amd64.AppImage
target/release/bundle/deb/Transcriptor_1.0.0_amd64.deb
target/release/bundle/rpm/Transcriptor-1.0.0-1.x86_64.rpm
```

## Verify before you publish

Check the declared dependencies actually made it into the packages — this is
the step that catches a package which installs and then fails to start:

```sh
dpkg -I target/release/bundle/deb/*.deb | grep -i depends
rpm -qp --requires target/release/bundle/rpm/*.rpm
```

The deb should list `libwebkit2gtk-4.1-0` and `libgtk-3-0`; Tauri generates
those itself, which is why `bundle.linux.deb.depends` is deliberately unset
here. The rpm dependency names are spelled differently per distribution
(`webkit2gtk4.1`, `gtk3`), so those are set explicitly under
`bundle.linux.rpm.depends`.

Then actually launch it, on a machine other than the one that built it if you
can:

```sh
# AppImage
chmod +x target/release/bundle/appimage/*.AppImage
./target/release/bundle/appimage/*.AppImage

# deb
sudo dpkg -i target/release/bundle/deb/*.deb

# rpm
sudo rpm -i target/release/bundle/rpm/*.rpm
```

## A note on the icon

The Linux config overrides `bundle.icon` to drop `icon.ico` and add the 512×512
`icon.png`. The base Windows list has no 512px entry, and the Linux bundlers
want the largest PNG they can find for the desktop entry and AppDir.

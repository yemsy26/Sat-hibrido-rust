//! Optional native bridges. `gpu` alone never invokes a native compiler.
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn joined_paths(paths: impl IntoIterator<Item = PathBuf>) -> OsString {
    env::join_paths(paths).expect("native build paths must be valid Windows paths")
}

fn required(path: &Path) {
    assert!(
        path.is_file(),
        "native bridge build dependency missing: {}",
        path.display()
    );
}

fn first_file(root: &Path, folders: &[&str], names: &[&str]) -> PathBuf {
    for folder in folders {
        for name in names {
            let path = root.join(folder).join(name);
            if path.is_file() {
                return path;
            }
        }
    }
    panic!(
        "TensorRT SDK dependency {} missing under {}",
        names.join(" or "),
        root.display()
    );
}

/// TENSORRT_ROOT manda; si falta, la version mas nueva en C:\\SDK y despues C:\\TensorRT.
fn default_tensorrt() -> Option<PathBuf> {
    let installed = std::fs::read_dir(r"C:\SDK")
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            let version: Vec<u32> = name
                .strip_prefix("TensorRT-")?
                .split('.')
                .map(str::parse)
                .collect::<Result<_, _>>()
                .ok()?;
            Some((version, Path::new(r"C:\SDK").join(name)))
        })
        .max()
        .map(|(_, root)| root);
    installed
        .into_iter()
        .chain([PathBuf::from(r"C:\TensorRT")])
        .find(|root| root.join("include/NvInfer.h").is_file())
}

const CUDA_TOOLKITS: &str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";

/// Newest MSVC x64 toolset among all Visual Studio installs (Community, Build
/// Tools...). Component IDs differ between editions, so the toolset is probed
/// on disk instead of filtering vswhere by component.
fn default_msvc() -> PathBuf {
    let vswhere =
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    required(&vswhere);
    let output = Command::new(&vswhere)
        .args(["-all", "-products", "*", "-property", "installationPath"])
        .output()
        .expect("failed to run vswhere");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|install| {
            let install = Path::new(install.trim());
            let version = std::fs::read_to_string(
                install.join(r"VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt"),
            )
            .ok()?;
            let toolset = install.join(r"VC\Tools\MSVC").join(version.trim());
            toolset.join(r"bin\Hostx64\x64\cl.exe").is_file().then(|| {
                let key: Vec<u32> = version
                    .trim()
                    .split('.')
                    .filter_map(|n| n.parse().ok())
                    .collect();
                (key, toolset)
            })
        })
        .max()
        .map(|(_, toolset)| toolset)
        .expect("no Visual Studio with MSVC x64 tools found; set RAMSEY43_MSVC_ROOT")
}

/// Minor toolset number: 14.4x is VS2022, 14.5x is VS2026.
fn msvc_minor(compiler: &Path) -> u32 {
    let name = compiler.file_name().unwrap_or_default().to_string_lossy();
    let mut parts = name.split('.');
    match (
        parts.next(),
        parts.next().and_then(|minor| minor.parse().ok()),
    ) {
        (Some("14"), Some(minor)) => minor,
        _ => panic!("unrecognized MSVC toolset folder: {}", compiler.display()),
    }
}

/// CUDA 12.6 only accepts VS2022 hosts; VS2026 needs the newest installed CUDA 13.x.
fn default_cuda(msvc_minor: u32) -> PathBuf {
    let toolkits = Path::new(CUDA_TOOLKITS);
    if msvc_minor < 50 {
        return toolkits.join("v12.6");
    }
    std::fs::read_dir(toolkits)
        .expect("CUDA toolkits folder missing")
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            let (major, minor) = name.strip_prefix('v')?.split_once('.')?;
            let version = (major.parse::<u32>().ok()?, minor.parse::<u32>().ok()?);
            (version.0 >= 13).then_some(version)
        })
        .max()
        .map(|(major, minor)| toolkits.join(format!("v{major}.{minor}")))
        .expect("MSVC 14.5x (VS2026) requires CUDA 13.x; install it or set RAMSEY43_CUDA_ROOT")
}

// TensorRT 11 aliases NV_TENSORRT_MAJOR to TRT_MAJOR_ENTERPRISE. Follow
// object-like integer macros instead of assuming the header contains "10".
fn header_integer(header: &str, name: &str) -> Option<u32> {
    let mut current = name.to_owned();
    for _ in 0..16 {
        let value = header.lines().find_map(|line| {
            let mut words = line.split_whitespace();
            if words.next() == Some("#define") && words.next() == Some(current.as_str()) {
                words.next()
            } else {
                None
            }
        })?;
        let value = value.trim_matches(['(', ')']);
        if let Ok(number) = value.parse() {
            return Some(number);
        }
        current = value.to_owned();
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mock = env::var_os("CARGO_FEATURE_NEURO_MOCK").is_some();
    let tensorrt = env::var_os("CARGO_FEATURE_NEURO_TRT").is_some();
    if !mock && !tensorrt {
        return;
    }
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT");
    let tensor_sdk = if tensorrt {
        Some(env::var_os("TENSORRT_ROOT").map(PathBuf::from).or_else(default_tensorrt)
            .expect("No se encuentra TensorRT. Instala y extrae el SDK NVIDIA TensorRT 10.x o 11.x para Windows x64 en C:\\SDK o apunta TENSORRT_ROOT a su raiz (include/ y lib/). Las ruedas de Python no incluyen el SDK C++."))
    } else {
        None
    };
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("windows"),
        "native risk bridges currently target Windows x64 only"
    );
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
        Ok("x86_64"),
        "native risk bridges require the Windows x64 ABI"
    );
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ENV").as_deref(),
        Ok("msvc"),
        "native risk bridges require the Windows MSVC ABI"
    );
    for name in [
        "RAMSEY43_CUDA_ROOT",
        "RAMSEY43_MSVC_ROOT",
        "RAMSEY43_WINDOWS_SDK_ROOT",
        "RAMSEY43_WINDOWS_SDK_VERSION",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-changed=src/ramsey43_risk_bridge.cu");
    println!("cargo:rerun-if-changed=src/ramsey43_risk_bridge.h");
    // Visual Studio updates replace the toolset folder, so both roots are
    // detected unless pinned explicitly; the pair is validated either way.
    let compiler = env::var_os("RAMSEY43_MSVC_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(default_msvc);
    let msvc_minor = msvc_minor(&compiler);
    let cuda = env::var_os("RAMSEY43_CUDA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_cuda(msvc_minor));
    let cuda_header = std::fs::read_to_string(cuda.join("include/cuda.h"))
        .unwrap_or_else(|_| panic!("cuda.h missing under {}", cuda.display()));
    let cuda_version = header_integer(&cuda_header, "CUDA_VERSION")
        .expect("failed to resolve CUDA_VERSION in cuda.h");
    assert!(
        match cuda_version / 1000 {
            12 => cuda_version >= 12060 && (30..50).contains(&msvc_minor),
            13 => msvc_minor >= 30,
            _ => false,
        },
        "CUDA {cuda_version} cannot host MSVC 14.{msvc_minor}: CUDA 12.6+ needs VS2022 (14.3x/14.4x), VS2026 (14.5x) needs CUDA 13.x"
    );
    println!(
        "cargo:warning=risk bridge: CUDA {cuda_version} at {}; MSVC {}",
        cuda.display(),
        compiler.display()
    );
    let sdk = PathBuf::from(
        env::var_os("RAMSEY43_WINDOWS_SDK_ROOT")
            .unwrap_or_else(|| OsString::from(r"C:\Program Files (x86)\Windows Kits\10")),
    );
    let sdk_version =
        env::var("RAMSEY43_WINDOWS_SDK_VERSION").unwrap_or_else(|_| "10.0.26100.0".into());
    let compiler_bin = compiler.join("bin/Hostx64/x64");
    let nvcc = cuda.join("bin/nvcc.exe");
    required(&nvcc);
    required(&compiler_bin.join("cl.exe"));
    required(&compiler_bin.join("link.exe"));
    required(&sdk.join(format!("Include/{sdk_version}/ucrt/stdio.h")));
    required(&sdk.join(format!("Lib/{sdk_version}/um/x64/kernel32.lib")));
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut path = vec![compiler_bin.clone(), cuda.join("bin")];
    path.extend(env::split_paths(&env::var_os("PATH").unwrap_or_default()));
    let includes = joined_paths([
        compiler.join("include"),
        sdk.join(format!("Include/{sdk_version}/ucrt")),
        sdk.join(format!("Include/{sdk_version}/shared")),
        sdk.join(format!("Include/{sdk_version}/um")),
        sdk.join(format!("Include/{sdk_version}/winrt")),
    ]);
    let libraries = joined_paths([
        compiler.join("lib/x64"),
        sdk.join(format!("Lib/{sdk_version}/ucrt/x64")),
        sdk.join(format!("Lib/{sdk_version}/um/x64")),
        cuda.join("lib/x64"),
    ]);
    let compile = |kind: &str, definition: &str, trt: Option<(&Path, u32)>| {
        let dll = output.join(format!("ramsey43_risk_{kind}.dll"));
        let mut command = Command::new(&nvcc);
        command
            .current_dir(&output)
            .env("PATH", joined_paths(path.clone()))
            .env("INCLUDE", &includes)
            .env("LIB", &libraries)
            .env("VCToolsInstallDir", &compiler)
            .args(["--shared", "--std=c++17", "-O2", "--cudart=static"])
            // PTX compilation does not query a GPU or execute CUDA work.
            // CUDA 13 dropped sm_70 and TensorRT 11 needs sm_75+, so both toolkits use 75.
            .arg("--generate-code=arch=compute_75,code=compute_75")
            .arg("--compiler-bindir")
            .arg(&compiler_bin)
            .arg(format!("-D{definition}=1"))
            .arg("-I")
            .arg(cuda.join("include"))
            .arg("-L")
            .arg(cuda.join("lib/x64"))
            .arg("-lcuda");
        if let Some((trt, major)) = trt {
            command.arg("-I").arg(trt.join("include"));
            for stem in ["nvinfer", "nvonnxparser"] {
                let versioned = format!("{stem}_{major}.lib");
                let unversioned = format!("{stem}.lib");
                command.arg(first_file(
                    trt,
                    &["lib", "lib/x64", "lib64"],
                    &[&versioned, &unversioned],
                ));
            }
        }
        let result = command
            .arg(manifest.join("src/ramsey43_risk_bridge.cu"))
            .arg("-o")
            .arg(&dll)
            .output()
            .expect("failed to start nvcc");
        assert!(
            result.status.success(),
            "CUDA {kind} bridge compilation failed:\n{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        required(&dll);
        dll
    };
    if mock {
        let dll = compile("mock", "RAMSEY43_RISK_MOCK_ONLY", None);
        println!("cargo:rustc-env=RAMSEY43_RISK_DLL={}", dll.display());
        println!("cargo:rustc-env=RAMSEY43_RISK_MOCK_DLL={}", dll.display());
    }
    if let Some(trt) = tensor_sdk {
        for name in ["NvInfer.h", "NvInferVersion.h", "NvOnnxParser.h"] {
            required(&trt.join("include").join(name));
        }
        let version = std::fs::read_to_string(trt.join("include/NvInferVersion.h"))
            .expect("failed to read TensorRT SDK version");
        let major = header_integer(&version, "NV_TENSORRT_MAJOR")
            .expect("failed to resolve NV_TENSORRT_MAJOR in NvInferVersion.h");
        assert!(
            matches!(major, 10 | 11),
            "neuro-trt supports the TensorRT 10.x and 11.x nvinfer SDKs; found major version {major} (TensorRT-RTX is a different API)"
        );
        let runtime_versioned = format!("nvinfer_{major}.dll");
        let parser_versioned = format!("nvonnxparser_{major}.dll");
        let runtime = first_file(
            &trt,
            &["lib", "bin", "lib/x64", "lib64"],
            &[&runtime_versioned, "nvinfer.dll"],
        );
        first_file(
            &trt,
            &["lib", "bin", "lib/x64", "lib64"],
            &[&parser_versioned, "nvonnxparser.dll"],
        );
        println!(
            "cargo:rerun-if-changed={}",
            trt.join("include/NvInferVersion.h").display()
        );
        let dll = compile("trt", "RAMSEY43_RISK_TENSORRT", Some((&trt, major)));
        println!("cargo:rustc-env=RAMSEY43_RISK_TRT_DLL={}", dll.display());
        println!("cargo:rustc-env=RAMSEY43_TRT_DLL={}", dll.display());
        println!(
            "cargo:rustc-env=RAMSEY43_TENSORRT_DLL_DIR={}",
            runtime.parent().unwrap().display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::header_integer;

    #[test]
    fn resolves_tensorrt_10_numeric_and_11_alias_versions() {
        assert_eq!(
            header_integer("#define NV_TENSORRT_MAJOR 10", "NV_TENSORRT_MAJOR"),
            Some(10)
        );
        assert_eq!(header_integer("#define TRT_MAJOR_ENTERPRISE 11\n#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE //!< version", "NV_TENSORRT_MAJOR"), Some(11));
    }

    #[test]
    fn rejects_missing_or_cyclic_version_definitions() {
        assert_eq!(
            header_integer("#define OTHER 11", "NV_TENSORRT_MAJOR"),
            None
        );
        assert_eq!(
            header_integer(
                "#define NV_TENSORRT_MAJOR ALIAS\n#define ALIAS NV_TENSORRT_MAJOR",
                "NV_TENSORRT_MAJOR"
            ),
            None
        );
    }
}

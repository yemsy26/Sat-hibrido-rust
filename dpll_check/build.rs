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
        Some(PathBuf::from(env::var_os("TENSORRT_ROOT")
            .expect("Falta la variable de entorno TENSORRT_ROOT. Instala y extrae el SDK NVIDIA TensorRT 10.x o 11.x para Windows x64 y apunta TENSORRT_ROOT a su raiz (include/ y lib/). Las ruedas de Python no incluyen el SDK C++.")))
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
    let cuda = PathBuf::from(env::var_os("RAMSEY43_CUDA_ROOT").unwrap_or_else(|| {
        OsString::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")
    }));
    let compiler = PathBuf::from(env::var_os("RAMSEY43_MSVC_ROOT").unwrap_or_else(|| {
        OsString::from(
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207",
        )
    }));
    assert!(
        compiler
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .starts_with("14.44."),
        "CUDA 12.6 bridge is pinned to MSVC 14.44 (VS2022)"
    );
    let cuda_header = std::fs::read_to_string(cuda.join("include/cuda.h"))
        .expect("CUDA 12.6 cuda.h must be installed");
    assert!(
        cuda_header
            .lines()
            .any(|line| line.split_whitespace().collect::<Vec<_>>()
                == ["#define", "CUDA_VERSION", "12060"]),
        "native risk bridges require CUDA toolkit 12.6, not a newer PATH compiler"
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
            .arg("--generate-code=arch=compute_70,code=compute_70")
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
            .expect("failed to start the pinned CUDA 12.6 nvcc");
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

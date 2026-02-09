# IOPDDL Solver

## Contents
- Source code
- Pre-compiled Linux binary (in `bin/`)
- Presentation slides (`slides.pdf`) from the ASPLOS Special Session (March 30, 2025)

## Build Instructions

### Prerequisites
- **GNU g++** compiler (version 11 or compatible)
- **CMake** build system

### Supported Platforms
- **Primary development:** Xubuntu 20.04 LTS (with g++-11)
- **MacOS development:** g++ from Homebrew (`brew install gcc`)
- Other platforms/compilers untested

### Building
```bash
mkdir build
cd build
cmake ..
make -j8
```

The compiled binary will be located at `build/iopddl`.

## Usage

### Command Line Options

```
Usage: ./iopddl <input_file_path> <timeout_in_seconds> [options]

Positional Arguments:
input_problem_path     Path to the input JSON file
timeout_in_seconds     Timeout duration in seconds

Options:
-h                     Show this help message and exit
-s <seed>              Set the seed value (default: 1)
-t <timeoutWCSP>       Set the timeout in seconds for internal WCSP solver (default: 12)
-j <numForks>          Set the number of forks for parallelism (default: 8)
-q                     Quiet mode (do not print node strategies)
```

### Example
```bash
./iopddl example.json 60 -s 42 -t 10 -j 4 -q
```

### Pre-compiled Binary
A statically compiled 64-bit x86 Linux binary is available in the `bin/` directory. Unzip the archive and, if needed, make the binary executable using:
```bash
chmod +x bin/iopddl
```

## Notes on Solver Behavior

The source code in this repository is almost identical to our original contest submission.
However, due to observed internal solver timeout issues with the instance `asplos-2025-iopddl-W.json`, we introduced the `-t` parameter.

To improve solver performance on that instance, try increasing the WCSP solver timeout:
```bash
./iopddl asplos-2025-iopddl-W.json 300 -t 60
```

We also fixed a usage constraint bug in our optimal algorithm, so you can expect lower-cost solutions on `asplos-2025-iopddl-V.json`.

## License

This project is licensed under the MIT License.
For full license details, see the file `LICENSE.md` in this repository.
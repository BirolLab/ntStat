project(
    'ntStat',
    'cpp',
    version: '1.1.1',
    license: 'GPL-3',
    default_options: [
        'cpp_std=c++17',
        'werror=false',
        'optimization=3',
        'b_lto=true',
    ],
)

conf = configuration_data()
conf.set('PROJECT_VERSION', meson.project_version())
configure_file(
    input: 'src/ntstat.py',
    output: 'ntstat',
    configuration: conf,
    install: true,
    install_dir: 'bin',
    install_mode: 'rwxr-xr-x',
)

py = import('python').find_installation('python3')

python3_conf = find_program('python3-config')
python3_include_path = run_command(python3_conf, ['--includes'], check: true).stdout().split('-I')[-1].strip()
python3_ldflags = run_command(python3_conf, ['--ldflags'], check: true).stdout().strip().split()
python_dep = declare_dependency(
    include_directories: include_directories(python3_include_path),
    link_args: python3_ldflags,
)

pybind11_conf = find_program('pybind11-config')
pybind11_include = run_command(pybind11_conf, ['--includes'], check: true).stdout().split('-I')[-1].strip()
pybind11 = declare_dependency(include_directories: [pybind11_include])

py_deps = [pybind11, python_dep]

subdir('src/modules/filter')
subdir('src/modules/hist')
subdir('src/modules/query')
subdir('src/modules/count')
project('an-ml-riddle', 'cpp',
        default_options : ['c_std=c23', 'cpp_std=c++23'],
        license : 'MPL-2.0',
        version : '0.1.0')

sources = []
subdir('schemas')
subdir('include')
subdir('src')

inc = include_directories('include')

deps = [schemas_dep,
        dependency('libsodium'),
        subproject('eigen').get_variable('eigen_dep')]

if host_machine.endian() != 'little'
  error('the host machine must be little-endian')
endif

# client_sources += shared_sources
# client_lib = library('anmlriddleclient', client_sources,
#                      include_directories : inc,
#                      dependencies : deps,
#                      version : meson.project_version(),
#                      install : true)

lib = library('anmlriddle', sources,
              include_directories : inc,
              dependencies : deps,
              version : meson.project_version(),
              install : true)

# dep = declare_dependency(
#         include_directories : inc,
#         link_with : [client_lib, server_lib]
# )

# subdir('tests')

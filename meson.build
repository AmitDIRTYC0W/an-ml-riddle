project('an-ml-riddle', 'cpp',
        default_options : ['c_std=c23', 'cpp_std=c++20'],
        license : 'MPL-2.0',
        version : '0.1.0')

shared_sources = []
subdir('schemas')
subdir('include')
subdir('src')

inc = include_directories('include')

deps = [schemas_dep,
        dependency('capnp'),
        dependency('kj'),
        dependency('libsodium')]

client_sources += shared_sources
client_lib = library('anmlriddleclient', client_sources,
                     include_directories : inc,
                     dependencies : deps,
                     version : meson.project_version(),
                     install : true)

server_sources += shared_sources
server_lib = library('anmlriddleserver', server_sources,
                     include_directories : inc,
                     dependencies : deps,
                     version : meson.project_version(),
                     install : true)

dep = declare_dependency(
        include_directories : inc,
        link_with : [client_lib, server_lib]
)

# subdir('tests')
